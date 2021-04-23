import torch
from torch.utils.data import Dataset
from .data.consts import *
from .utils import *


class GenderPerformrDataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __getitem__(self, index):
        username = prep_name(self.data[index])
        if len(username) > USERNAME_LEN:
            return torch.LongTensor(username[:USERNAME_LEN]), USERNAME_LEN
        else:
            username_len = len(username)
            return torch.cat([torch.LongTensor(username),
                              torch.zeros(USERNAME_LEN - username_len, dtype=torch.long)]), \
                   len(username)

    def __len__(self):
        return len(self.data)
