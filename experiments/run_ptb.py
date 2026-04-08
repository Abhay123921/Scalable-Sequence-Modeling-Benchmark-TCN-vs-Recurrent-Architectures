import torch
import math

from data.ptb import load_ptb, batchify, get_batch
from models.tcn.tcn_lm import TCN_LM
from models.lstm.lstm_lm import LSTM_LM
from models.rnn.rnn_lm import RNN_LM
from models.gru.gru_lm import GRU_LM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data, vocab = load_ptb("data/ptb.train.txt")
data = batchify(data, batch_size=20)

models = {
    "TCN": TCN_LM(len(vocab), [128]*10),
    "LSTM": LSTM_LM(len(vocab)),
    "GRU": GRU_LM(len(vocab)),
    "RNN": RNN_LM(len(vocab))   
}
criterion = torch.nn.CrossEntropyLoss()

seq_len = 50

for name, model in models.items():

    print(f"\n🚀 Training {name}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(5):

        total_loss = 0

        for i in range(0, data.size(1)-1, seq_len):

            x, y = get_batch(data, i, seq_len)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)

            loss = criterion(
                out.view(-1, len(vocab)),
                y.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            total_loss += loss.item()

        ppl = math.exp(total_loss / (data.size(1)//seq_len))
        print(f"{name} Epoch {epoch} | Perplexity: {ppl:.2f}")