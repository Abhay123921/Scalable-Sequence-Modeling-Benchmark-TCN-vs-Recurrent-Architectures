from models.tcn.tcn_base import TCN_MNIST
from models.lstm.lstm_lm import LSTM_Model
from models.rnn.rnn_base import RNN_Model
from models.gru.gru_lm import GRU_Model

models = {
    "TCN": TCN_MNIST(),
    "LSTM": LSTM_Model(1, 128, 10),
    "RNN": RNN_Model(1, 128, 10),
    "GRU": GRU_Model(1, 128, 10),
}

results = {}

for name, model in models.items():
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)

    acc = evaluate(model, test_loader, device)
    results[name] = acc
    print(f"{name}: {acc}")