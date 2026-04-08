# models/tcn_seq.py
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) \
            if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)[:, :, :-self.conv1.padding[0]]
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)[:, :, :-self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        return out + res


class TCNSeq(nn.Module):
    def __init__(self, vocab_size=10, emb_dim=16, channels=[16]*8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = emb_dim if i == 0 else channels[i-1]
            out_ch = channels[i]

            layers.append(TemporalBlock(in_ch, out_ch, 3, dilation))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], vocab_size)

    def forward(self, x):
        x = self.embedding(x)        # [B, L, E]
        x = x.transpose(1, 2)        # [B, E, L]

        y = self.network(x)          # [B, C, L]
        y = y.transpose(1, 2)        # [B, L, C]

        return self.fc(y)            # [B, L, vocab]