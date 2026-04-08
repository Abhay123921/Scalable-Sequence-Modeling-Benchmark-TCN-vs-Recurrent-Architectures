import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) \
            if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNSeq(nn.Module):
    def __init__(self, vocab_size, channels):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 32)

        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = 32 if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, channels[i], 3, dilation))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], vocab_size)

    def forward(self, x):
        x = self.embed(x)         # [B,T,E]
        x = x.transpose(1, 2)     # [B,E,T]

        y = self.network(x)

        y = y.transpose(1, 2)     # [B,T,C]
        return self.fc(y)