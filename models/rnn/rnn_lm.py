import torch.nn as nn

class RNN_LM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 128)

        self.rnn = nn.RNN(
            128,
            256,
            num_layers=2,
            nonlinearity='tanh',
            batch_first=True
        )

        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.rnn(x)
        return self.fc(out)