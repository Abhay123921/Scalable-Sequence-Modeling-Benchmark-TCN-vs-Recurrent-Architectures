import torch
import torch.nn as nn


# -----------------------------
# Causal Convolution Trim Layer
# -----------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


# -----------------------------
# Residual Block (TCN)
# -----------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)

        return self.final_relu(out + res)


# -----------------------------
# Full TCN Network
# -----------------------------
class TCNClassifier(nn.Module):
    def __init__(self, input_size, channels, num_classes, kernel_size=3):
        super().__init__()

        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else channels[i - 1]
            out_ch = channels[i]

            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation)
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        # x: [B, T, C] → [B, C, T]
        x = x.transpose(1, 2)

        y = self.network(x)

        # take last timestep
        y = y[:, :, -1]

        return self.fc(y)