import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class ChunkBasedConformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, ffn_dim, num_layers, kernel_size, chunk_size, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ffn_dim, kernel_size, chunk_size, dropout) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.dropout(x)

class ConformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, kernel_size, chunk_size, dropout=0.1):
        super().__init__()
        self.feed_forward1 = FeedForward(hidden_dim, ffn_dim, dropout)
        self.self_attn = ChunkBasedMultiHeadAttention(hidden_dim, num_heads, chunk_size, dropout)
        self.conv = ChunkBasedConvModule(hidden_dim, kernel_size, chunk_size)
        self.feed_forward2 = FeedForward(hidden_dim, ffn_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.dropout(self.feed_forward1(x))
        x = x + self.dropout(self.self_attn(x))
        x = x + self.dropout(self.conv(x))
        x = x + 0.5 * self.dropout(self.feed_forward2(x))
        return self.norm(x)

class ChunkBasedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, chunk_size, dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
    def forward(self, x):
        attn_mask = self.get_chunk_mask(x.size(1)).to(x.device)
        return self.attention(x, x, x, attn_mask=attn_mask)[0]

    def get_chunk_mask(self, seq_len):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(0, seq_len, self.chunk_size):
            mask[i:i+self.chunk_size, i:i+self.chunk_size] = 0
        return mask.triu(1)

class ChunkBasedConvModule(nn.Module):
    def __init__(self, hidden_dim, kernel_size, chunk_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2, groups=hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.SiLU()
        self.pointwise_conv = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv(x)
        return x.transpose(1, 2)

class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class StreamSpeech(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ChunkBasedConformer(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            kernel_size=config.kernel_size,
            chunk_size=config.chunk_size,
            dropout=config.dropout
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=config.hidden_dim, 
                                       nhead=config.num_heads, 
                                       dim_feedforward=config.ffn_dim, 
                                       dropout=config.dropout), 
            num_layers=config.num_layers
        )
        self.t2u_generator = nn.Linear(config.hidden_dim, config.vocab_size)
        self.asr_ctc = nn.Linear(config.hidden_dim, config.vocab_size)
        self.s2tt_ctc = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(self, src, tgt=None):
        encoder_output = self.encoder(src)
        asr_ctc_output = self.asr_ctc(encoder_output)
        s2tt_ctc_output = self.s2tt_ctc(encoder_output)
        
        if tgt is not None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
            decoder_output = self.decoder(tgt, encoder_output, tgt_mask=tgt_mask)
            t2u_output = self.t2u_generator(decoder_output)
            return asr_ctc_output, s2tt_ctc_output, decoder_output, t2u_output 
        else:
            return asr_ctc_output, s2tt_ctc_output

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class AdvancedDataProcessor:
    def __init__(self, config):
        self.config = config
        self.mel_basis = librosa.filters.mel(**config.mel)
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(config.wav2vec_model)
        self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(config.wav2vec_model)

    def process(self, waveform):
        waveform = self._preprocess_audio(waveform)
        mel = self._compute_mel_spectrogram(waveform)
        wav2vec_features = self._extract_wav2vec_features(waveform)
        return mel, wav2vec_features

    def _preprocess_audio(self, waveform):
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.size(1) / self.config.sample_rate != waveform.size(1) / 16000:
            waveform = torchaudio.transforms.Resample(16000, self.config.sample_rate)(waveform)
        return waveform

    def _compute_mel_spectrogram(self, waveform):
        spec = torch.stft(waveform, **self.config.stft, return_complex=True)
        spec = torch.abs(spec)
        mel = torch.matmul(self.mel_basis.to(spec.device), spec)
        return torch.log(torch.clamp(mel, min=1e-5))

    def _extract_wav2vec_features(self, waveform):
        inputs = self.wav2vec_processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=self.config.sample_rate)
        with torch.no_grad():
            return self.wav2vec_model(**inputs).last_hidden_state