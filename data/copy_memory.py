import torch
import numpy as np

def generate_copy_memory(batch_size=32, T=500, seq_len=10):

    total_len = T + 2 * seq_len

    x = np.zeros((batch_size, total_len), dtype=int)
    y = np.zeros((batch_size, total_len), dtype=int)

    for i in range(batch_size):

        seq = np.random.randint(1, 9, size=seq_len)

        # input
        x[i, :seq_len] = seq
        x[i, seq_len + T] = 9   # delimiter

        # target
        y[i, -seq_len:] = seq

    return torch.LongTensor(x), torch.LongTensor(y)