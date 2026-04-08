# experiments/run_seq_mnist.py

import torch
from data.sequential_mnist import get_seq_mnist
from models.tcn.tcn_classifier import TCNClassifier
from models.lstm.lstm_classifier import LSTMClassifier
from models.gru.gru_classifier import GRUClassifier
from models.rnn.rnn_classifier import RNNClassifier
from training.trainer import train, evaluate

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
train_loader, test_loader = get_seq_mnist()

# define models
models = {
    "RNN": RNNClassifier(1, 512, 10),
    "LSTM": LSTMClassifier(1, 256, 10),
    "GRU": GRUClassifier(1, 128, 10),
    "TCN": TCNClassifier(1, [25]*8, 10)
}

# training loop
for name, model in models.items():

    print(f"\n🚀 Training {name}")

    model = model.to(device)

    if name == "TCN":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    if name == "RNN":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.007)  
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):

        loss = train(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)

        print(f"{name} Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")