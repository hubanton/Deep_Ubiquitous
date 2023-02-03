from Dataloading import create_windows, get_patient_data
from train_and_evaluate import rnn_training
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.load('X_data.pt')
y = torch.load('y_data.pt')

print(X.shape)
print(y.shape)

rnn_training(X, y, 'Test Training', device, torch.nn.RNN, epochs=5)

# window_sizes = [400, 1200, 2000, 4000]
# hop_sizes = [80, 240, 400, 800]
#
# for win_size, hop_size in zip(window_sizes, hop_sizes):
#
#     X = torch.load(f'X_data_ws_{win_size}_hs_{hop_size}.pt')
#     y = torch.load(f'y_data_ws_{win_size}_hs_{hop_size}.pt')
#
#     rnn_training(X, y, f'Window size: {win_size}', device, torch.nn.RNN)

# rnn_training(X, y, 'RNN', device, torch.nn.RNN)
#
# rnn_training(X, y, 'LSTM', device, torch.nn.LSTM)
#
# rnn_training(X, y, 'GRU', device, torch.nn.GRU)
