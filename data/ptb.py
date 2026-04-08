# data/ptb.py

import torch

def load_ptb(path):
    text = open(path).read().replace("\n", "<eos>").split()

    vocab = {w: i for i, w in enumerate(set(text))}
    data = torch.LongTensor([vocab[w] for w in text])

    return data, vocab


def batchify(data, batch_size):
    n_batch = data.size(0) // batch_size
    data = data[:n_batch * batch_size]

    # shape → [B, T]
    data = data.view(batch_size, -1)

    return data


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, source.size(1) - 1 - i)

    x = source[:, i:i+seq_len]
    y = source[:, i+1:i+seq_len+1]

    return x, y