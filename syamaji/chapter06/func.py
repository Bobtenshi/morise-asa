import numpy as np


def MyBlackman(N):
    n = np.arange(N, dtype=float)
    win = (
        0.42 - 0.5 * np.cos(2* np.pi * n / (N-1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    )
    return win

def MinimumPhase(x):
    x_len_half = int(len(x)/2 - 1)
    X = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(x[:])))))
    w = np.block([1, 2 * np.ones(x_len_half), 1, np.zeros(x_len_half)])
    y = np.real(np.fft.ifft(np.exp(np.fft.fft(w*X))))
    return y
