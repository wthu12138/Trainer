from torch.utils.data import Dataset
import torch


class ToyDataset(Dataset):

    def __init__(self, length):
        self.length = length
        self.data = torch.randn(self.length,
                                10,
                                generator=torch.Generator().manual_seed(42))
        self.label = torch.randn(self.length,
                                 1,
                                 generator=torch.Generator().manual_seed(42))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.data[item], self.label[item]
