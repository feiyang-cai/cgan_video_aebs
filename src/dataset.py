import torch
from torch.utils.data import Dataset, DataLoader

class AEBSVideoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] # (seq_len, 3, 32, 32)

class DummyDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __len__(self):
        return 32

    def __getitem__(self, idx):
        data = torch.randn(self.seq_len, 3, 32, 32)
        labels = torch.randn(self.seq_len)
        return data, labels