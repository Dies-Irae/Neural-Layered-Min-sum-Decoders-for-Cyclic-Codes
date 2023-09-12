import numpy as np


class LinearCode:
    def __init__(self, family, n, k, G, H, d=None):
        """
        :param n: code length, integer
        :param k: bit length before encoding, integer
        :param G: k*n generate matrix, numpy array
        :param H: (n-k)*n parity check matrix, numpy array
        :param d: minimum Hamming distance
        """
        self.family = family
        self.n = n
        self.k = k
        self.G = G
        self.H = H
        self.d = d

    def encode(self, message):
        """
        Input: origin message\n
        Return: encoded codeword
        """
        return np.remainder(np.matmul(message, self.G), 2)

    def generateBatch(self, BatchSize):
        """
        return a randomly generated message in NumPy array form, which size is(infomation bits length(k), BatchSize)
        """
        batch = np.random.randint(2, size=(BatchSize, self.k))
        return batch

    def AWGN(self, inputs, SNR):
        """
        inputs: codewords to be send\n
        SNR: range of signal noise ratio(dB) for example: (0,7)
        """
        inputs = -(inputs * 2 - 1)  # BPSK modulation
        SNRArray = np.random.uniform(SNR[0], SNR[1], inputs.shape[0])
        sigmaArray = np.sqrt(1 / (2 * self.k / self.n * np.power(10, SNRArray / 10)))
        noise = np.random.randn(inputs.shape[1], inputs.shape[0]) * sigmaArray
        noise = noise.T
        inputs = inputs + noise
        return inputs, sigmaArray
