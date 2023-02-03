import random
import numpy as np
from Dataloading import create_windows, get_patient_data, add_features
from train_and_evaluate import rnn_training
import torch
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed for reproducible results ( for 1 f) )
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# set this environment variable for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"

window_performances = []

window_sizes = [200, 200, 400, 400, 800, 800]
hop_sizes = [20, 40, 40, 80, 80, 160]

for win_size, hop_size in zip(window_sizes, hop_sizes):
    X = torch.load(f'data/X_data_ws_{win_size}_hs_{hop_size}.pt')
    y = torch.load(f'data/y_data_ws_{win_size}_hs_{hop_size}.pt')

    performance = rnn_training(X, y, f'Window size: {win_size}, Hop size: {hop_size}', device, torch.nn.RNN, epochs=100)

    window_performances.append(max(performance['macro']))

best_idx = np.argmax(np.array(window_performances))

print(f"Best window size and hop size: {window_sizes[best_idx]}, {hop_sizes[best_idx]}")

X = torch.load(f'X_data_ws_{window_sizes[best_idx]}_hs_{hop_sizes[best_idx]}.pt')
y = torch.load(f'y_data_ws_{window_sizes[best_idx]}_hs_{hop_sizes[best_idx]}.pt')

feat_performances = []

for use_acc_features in [True, False]:
    for use_freq_domain_features in [True, False]:
        new_X = add_features(X, add_acceleration_features=use_acc_features,
                             add_freq_domain_features=use_freq_domain_features)

        performance = rnn_training(new_X, y, f'acc_f: {use_acc_features}, freq_f: {use_freq_domain_features}', device, torch.nn.RNN)

        feat_performances.append(max(performance['macro']))

best_tuple = [(True, True), (True, False), (False, True), (False, False)][np.argmax(np.array(feat_performances))]

print(f"best combination: {best_tuple}")

best_X = add_features(X, add_acceleration_features=best_tuple[0], add_freq_domain_features=best_tuple[1])

models = [torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU]
model_names = ['RNN', 'LSTM', 'GRU']

model_performances = []

for model, model_name in zip(models, model_names):
    performance = rnn_training(X, y, model_name, device, model)

    model_performances.append(max(performance['macro']))

best_model = model_names[np.argmax(np.array(model_performances))]

print(f"best model: {best_model}")
