from matplotlib import pyplot as plt
from Model import MyRNN
from Dataloading import get_patient_data
import numpy as np
import pandas as pd
import torch

X, y = get_patient_data(use_interpolation=True)

model = MyRNN(in_features=7, hidden_features=10, num_layers=2)

print(X)
print(y)

window_size = 10
hop_size = window_size//2
output = torch.tensor([]) # shape [num_files*num_sliding_widows, window_size, features]
labels = []
for file in X["file_name"].unique():
    features_df = X[X['file_name'] == file]
    labels_df = y[y['file_name'] == file].values
    last_frame = np.max(X[X['file_name'] == file]["frame_id"])
    for pivot in range(0, last_frame - window_size, hop_size):
        labels.append(labels_df[pivot+window_size-1][0])
        window = torch.tensor(features_df[features_df["frame_id"].between(pivot, pivot + window_size, inclusive="left")].drop(["frame_id", "file_name"], axis=1).values)
        output = torch.cat((output, torch.unsqueeze(window, dim=0)), dim=0)
labels = torch.tensor(labels)
print(labels)
print(labels.shape)
print(output)
print(output.shape)
