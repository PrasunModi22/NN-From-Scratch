import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import logsumexp
from keras import datasets

class MLP():
    def __init__(self, din, dout):
        self.W = (np.random.randn(din, dout) * 2 - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.n = (np.random.randn(din) * 2 - 1) * (np.sqrt(6) / np.sqrt(din + dout))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    