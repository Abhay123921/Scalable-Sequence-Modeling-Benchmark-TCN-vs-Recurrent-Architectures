import torch
from data.copy_memory import generate_copy_memory

from models.tcn.tcn_seq import TCNSeq
from models.lstm.lstm_seq import LSTMSeq
from models.gru.gru_seq import GRUSeq
from models.rnn.rnn_seq import RNNSeq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "TCN": TCNSeq(10, [32]*10),
    "LSTM": LSTMSeq(10),
    "GRU": GRUSeq(10),
    "RNN": RNNSeq(10)
}

criterion = torch.nn.CrossEntropyLoss()

for name, model in models.items():

    print(f"\n🚀 Training {name}")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    for step in range(1001):

        x, y = generate_copy_memory(batch_size=32, T=500)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)   # [B,T,V]

        # only evaluate last seq_len positions
        seq_len = 10

        out_last = out[:, -seq_len:, :]     # [B,10,V]
        y_last = y[:, -seq_len:]            # [B,10]

        loss = criterion(
            out_last.reshape(-1, 10),
            y_last.reshape(-1))
        
        preds = out_last.argmax(dim=-1)
        acc = (preds == y_last).float().mean().item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step % 250 == 0:
            print(f"{name} Step {step} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")