import torch
import os

# Create a directory to store the .pt files
os.makedirs('dummy_train', exist_ok=True)

# Create and save dummy .pt files
for i in range(5):
    dummy_mel = torch.randn(80, 200)  # Random mel spectrogram
    dummy_wav2vec = torch.randint(0, 10000, (200,))  # Random wav2vec features
    torch.save({'mel': dummy_mel, 'wav2vec': dummy_wav2vec}, f'dummy_train/data_{i}.pt')

print("Dummy .pt files created and saved in 'dummy_train' directory.")
