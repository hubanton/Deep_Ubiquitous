import os
import random

import numpy as np
import torch

from data_loading import add_features, create_windows, get_patient_data
from train_and_evaluate import rnn_training

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set seed for reproducible results ( for 1 f) )
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# set this environment variable for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"

num_features = 7
num_acc_features = 3
num_frequency_features = 12

window_performances = []

window_sizes = [200, 200, 400, 400, 800, 800]
hop_sizes = [20, 40, 40, 80, 80, 160]

X, y = get_patient_data(use_interpolation=True)

for win_size, hop_size in zip(window_sizes, hop_sizes):
    x_data_path = f'data/X_data_ws_{win_size}_hs_{hop_size}.pt'
    y_data_path = f'data/y_data_ws_{win_size}_hs_{hop_size}.pt'
    if not os.path.exists(x_data_path) or not os.path.exists(y_data_path):
        create_windows(X, y, win_size, hop_size)
    X = torch.load(f'data/X_data_ws_{win_size}_hs_{hop_size}.pt')
    y = torch.load(f'data/y_data_ws_{win_size}_hs_{hop_size}.pt')

    performance = rnn_training(X, y, f'Window size {win_size}, Hop size {hop_size}', device, torch.nn.RNN, epochs=100)

    print(performance)

    window_performances.append(max(performance['macro']))

best_idx = np.argmax(np.array(window_performances))

print(f"Best window size and hop size: {window_sizes[best_idx]}, {hop_sizes[best_idx]}")

X = torch.load(f'data/X_data_ws_{window_sizes[best_idx]}_hs_{hop_sizes[best_idx]}.pt')
y = torch.load(f'data/y_data_ws_{window_sizes[best_idx]}_hs_{hop_sizes[best_idx]}.pt')

feat_performances = []

for use_acc_feat in [True, False]:
    for use_freq_feat in [True, False]:
        new_X = add_features(X, add_acceleration_features=use_acc_feat,
                             add_freq_domain_features=use_freq_feat)

        input_size = num_features + (num_acc_features * use_acc_feat) + (num_frequency_features * use_freq_feat)

        performance = rnn_training(new_X, y, f'acc_f {use_acc_feat}, freq_f {use_freq_feat}', device,
                                   torch.nn.RNN, in_features=input_size)

        feat_performances.append(max(performance['macro']))

best_tuple = [(True, True), (True, False), (False, True), (False, False)][np.argmax(np.array(feat_performances))]

print(f"best combination: {best_tuple}")

best_X = add_features(X, add_acceleration_features=best_tuple[0], add_freq_domain_features=best_tuple[1])

input_size = num_features + num_acc_features * best_tuple[0] + num_frequency_features * best_tuple[1]

models = [torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU]
model_names = ['RNN', 'LSTM', 'GRU']

model_performances = []

for model, model_name in zip(models, model_names):
    performance = rnn_training(best_X, y, model_name, device, model, in_features=input_size)

    model_performances.append(max(performance['macro']))

best_model = model_names[np.argmax(np.array(model_performances))]

print(f"best model: {best_model}")
