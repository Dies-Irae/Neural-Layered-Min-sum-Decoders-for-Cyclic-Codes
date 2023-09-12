from torch.utils.data import Dataset
import torch
import numpy as np
import math

class Trainset(Dataset):
    def __init__(self, code, size, SNR, device, use_zero_codeword=True):
        if use_zero_codeword:
            self.labels = np.zeros((size, code.n))
        else:
            self.labels = code.encode(code.generateBatch(size))
        np.random.seed(0)
        self.samples, self.sigma = code.AWGN(self.labels, SNR)
        self.sigma = torch.from_numpy(self.sigma).float()
        self.sigma = torch.unsqueeze(self.sigma, 1)
        self.samples = torch.from_numpy(self.samples).float()
        self.labels = torch.from_numpy(self.labels).float()

        self.sigma = self.sigma.to(device)
        self.labels = self.labels.to(device)
        self.samples = self.samples.to(device)


class TestSet(Dataset):
    def __init__(self, code, size, SNR, device):
        message = code.generateBatch(size)
        self.labels = code.encode(message)
        self.samples = torch.from_numpy(code.AWGN(self.labels, (SNR, SNR))[0])
        self.labels = torch.from_numpy(self.labels).float()
        self.labels = self.labels.to(device)
        self.samples = self.samples.to(device)
