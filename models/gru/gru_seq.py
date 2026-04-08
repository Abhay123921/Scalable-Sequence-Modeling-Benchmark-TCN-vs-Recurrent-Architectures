import torch.nn as nn

class GRUSeq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.gru(x)
        return self.fc(out)