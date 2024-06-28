import os
import json
import argparse
import logging
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import StreamSpeech, AdvancedDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    kernel_size: int
    chunk_size: int
    vocab_size: int
    upsampling_rate: int
    dropout: float

@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    model_save_path: str

@dataclass
class DataConfig:
    sample_rate: int
    mel: dict
    stft: dict
    wav2vec_model: str

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

# Embedded configuration
DEFAULT_CONFIG = {
    "model": {
        "input_dim": 200,
        "hidden_dim": 256,
        "num_heads": 4,
        "ffn_dim": 1024,
        "num_layers": 12,
        "kernel_size": 31,
        "chunk_size": 50,
        "vocab_size": 10000,
        "upsampling_rate": 25,
        "dropout": 0.1
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "model_save_path": "streamspeech_model.pth"
    },
    "data": {
        "sample_rate": 16000,
        "mel": {
            "n_mels": 80,
            "n_fft": 400,
            "hop_length": 160,
            "fmin": 0,
            "fmax": 8000
        },
        "stft": {
            "n_fft": 400,
            "hop_length": 160,
            "win_length": 400,
            "window": "hann"
        },
        "wav2vec_model": "facebook/wav2vec2-base-960h"
    }
}

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use default."""
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = DEFAULT_CONFIG
    return Config(
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data'])
    )

class SpeechDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.load(os.path.join(self.data_dir, self.file_names[idx]))
        return data['mel'], data['wav2vec']

def prepare_data(config: Config, input_dir: str, output_dir: str) -> None:
    """Prepare data for training."""
    processor = AdvancedDataProcessor(config.data)
    os.makedirs(output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(input_dir), desc="Processing audio files"):
        if file_name.endswith('.wav'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace('.wav', '.pt'))

            waveform, _ = torchaudio.load(input_path)
            mel_spectrogram, wav2vec_features = processor.process(waveform)

            torch.save({'mel': mel_spectrogram, 'wav2vec': wav2vec_features}, output_path)

def train(config: Config, model: nn.Module, data_dir: str) -> None:
    """Train the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = SpeechDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    ce_loss = nn.CrossEntropyLoss(ignore_index=0)

    model.to(device)

    best_loss = float('inf')
    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0.0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.training.epochs}") as pbar:
            for mel, wav2vec in pbar:
                mel, wav2vec = mel.to(device), wav2vec.to(device)

                optimizer.zero_grad()

                asr_ctc_output, s2tt_ctc_output, decoder_output, t2u_output = model(mel, wav2vec)

                loss = sum([
                    ctc_loss(asr_ctc_output.transpose(0, 1), wav2vec, [wav2vec.size(1)] * wav2vec.size(0), [asr_ctc_output.size(1)] * asr_ctc_output.size(0)),
                    ctc_loss(s2tt_ctc_output.transpose(0, 1), wav2vec, [wav2vec.size(1)] * wav2vec.size(0), [s2tt_ctc_output.size(1)] * s2tt_ctc_output.size(0)),
                    ce_loss(decoder_output.view(-1, decoder_output.size(-1)), wav2vec.view(-1)),
                    ce_loss(t2u_output.view(-1, t2u_output.size(-1)), wav2vec.view(-1))
                ])

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{config.training.epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.training.model_save_path)
            logger.info(f"New best model saved to {config.training.model_save_path}")

def inference(config: Config, model: nn.Module, input_file: str, processor: AdvancedDataProcessor) -> np.ndarray:
    """Run inference on input file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    waveform, _ = torchaudio.load(input_file)
    mel_spectrogram, wav2vec_features = processor.process(waveform)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)
    wav2vec_features = wav2vec_features.unsqueeze(0).to(device)

    with torch.no_grad():
        asr_ctc_output, s2tt_ctc_output = model(mel_spectrogram, wav2vec_features)

    s2tt_output = torch.argmax(s2tt_ctc_output, dim=-1)

    return s2tt_output.squeeze().cpu().numpy()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="StreamSpeech Model")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', choices=['prepare', 'train', 'inference'], required=True,
                        help='Mode of operation: prepare data, train model, or run inference')
    parser.add_argument('--input_dir', help='Input directory for data preparation or inference')
    parser.add_argument('--output_dir', help='Output directory for data preparation or inference results')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return args

def main():
    """Main function to run the StreamSpeech model."""
    args = parse_arguments()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}. Using default configuration.")
        config = load_config()
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in config file: {args.config}")
        sys.exit(1)

    if args.mode == 'prepare':
        if not args.input_dir or not args.output_dir:
            logger.error("Both --input_dir and --output_dir are required for 'prepare' mode")
            sys.exit(1)
        prepare_data(config, args.input_dir, args.output_dir)
    elif args.mode == 'train':
        if not args.input_dir:
            logger.error("--input_dir is required for 'train' mode")
            sys.exit(1)
        model = StreamSpeech(config.model)
        train(config, model, args.input_dir)
    elif args.mode == 'inference':
        if not args.input_dir or not args.output_dir:
            logger.error("Both --input_dir and --output_dir are required for 'inference' mode")
            sys.exit(1)
        model = StreamSpeech(config.model)
        try:
            model.load_state_dict(torch.load(config.training.model_save_path))
        except FileNotFoundError:
            logger.error(f"Model file not found: {config.training.model_save_path}")
            sys.exit(1)
        processor = AdvancedDataProcessor(config.data)
        output = inference(config, model, args.input_dir, processor)
        output_path = os.path.join(args.output_dir, 'inference_output.npy')
        np.save(output_path, output)
        logger.info(f"Inference output saved to {output_path}")

if __name__ == "__main__":
    main()