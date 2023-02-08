import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from scipy import interpolate
from scipy.fft import fft
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_patient_data(use_interpolation=False, binarize=False):
    step_size = 0.025

    s1_dir = sorted(glob(os.path.join("Datasets_Healthy_Older_People/S1_Dataset", "*")))
    s2_dir = sorted(glob(os.path.join("Datasets_Healthy_Older_People/S2_Dataset", "*")))

    patient_data = pd.DataFrame()

    for record in s1_dir:
        if use_interpolation:
            record_data = pd.read_csv(record, header=None)
            interpolated_df = pd.DataFrame()
            t_start = int(record_data[0].min())
            t_end = record_data[0].max()
            x_new = np.arange(t_start, t_end, step_size)
            interpolated_df[0] = x_new
            for i in range(1, 9):
                f = interpolate.interp1d(record_data[0], record_data[i])
                interpolated_df[i] = f(x_new)

            interpolated_df[8] = round(interpolated_df[8]).astype(int)

            interpolated_df['file_name'] = record.split(os.sep)[-1]
            interpolated_df['frame_id'] = interpolated_df.index
            patient_data = pd.concat((patient_data, interpolated_df), axis=0)
        else:
            record_data = pd.read_csv(record, header=None)
            record_data['file_name'] = record.split(os.sep)[-1]
            record_data['frame_id'] = record_data.index
            patient_data = pd.concat((patient_data, record_data), axis=0)

    for record in s2_dir:
        if use_interpolation:
            record_data = pd.read_csv(record, header=None)
            interpolated_df = pd.DataFrame()
            t_start = int(record_data[0].min())
            t_end = record_data[0].max()
            x_new = np.arange(t_start, t_end, step_size)
            interpolated_df[0] = x_new
            for i in range(1, 9):
                f = interpolate.interp1d(record_data[0], record_data[i])
                interpolated_df[i] = f(x_new)

            interpolated_df[8] = round(interpolated_df[8]).astype(int)

            interpolated_df['file_name'] = record.split(os.sep)[-1]
            interpolated_df['frame_id'] = interpolated_df.index
            patient_data = pd.concat((patient_data, interpolated_df), axis=0)
        else:
            record_data = pd.read_csv(record, header=None)
            record_data['file_name'] = record.split(os.sep)[-1]
            record_data['frame_id'] = record_data.index
            patient_data = pd.concat((patient_data, record_data), axis=0)

    if binarize:
        patient_data[8] = patient_data[8].replace([1, 2, 3], 0)
        patient_data[8] = patient_data[8].replace(4, 1)

    for i in [1, 2, 3, 5, 6, 7]:
        patient_data[i] = (patient_data[i] - patient_data[i].min()) / (patient_data[i].max() - patient_data[i].min())

    y = patient_data[[8, 'file_name', 'frame_id']]
    X = patient_data.drop(columns=[0, 8])

    return X, y


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.num_examples = len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

    def __len__(self):
        return self.num_examples


def get_dataloaders(X, y, batch_size=16, shuffle=True, drop_last=True):
    full_dataset = SequenceDataset(X, y)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=drop_last)

    val_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                drop_last=False)

    return train_dataloader, val_dataloader


def create_windows(X, y, window_size, hop_size):
    output = torch.tensor([])  # shape [num_files*num_sliding_widows, window_size, features]
    labels = []
    for file in tqdm(X["file_name"].unique()):
        features_df = X[X['file_name'] == file]
        labels_df = y[y['file_name'] == file].values
        last_frame = np.max(X[X['file_name'] == file]["frame_id"])
        for pivot in range(0, last_frame - window_size, hop_size):
            labels.append(labels_df[pivot + window_size - 1][0])
            window = torch.tensor(
                features_df[features_df["frame_id"].between(pivot, pivot + window_size, inclusive="left")].drop(
                    ["frame_id", "file_name"], axis=1).values)
            output = torch.cat((output, torch.unsqueeze(window, dim=0)), dim=0)

    labels = torch.tensor(labels) - 1
    output = output.to(torch.float32)

    torch.save(output, f'data/X_data_ws_{window_size}_hs_{hop_size}.pt')
    torch.save(labels, f'data/y_data_ws_{window_size}_hs_{hop_size}.pt')


def add_features(X, add_acceleration_features=True, add_freq_domain_features=True):
    if add_acceleration_features:
        phi = torch.unsqueeze(
            torch.arctan(torch.div(X[:, :, 0], torch.sqrt(torch.pow(X[:, :, 1], 2) + torch.pow(X[:, :, 2], 2)))),
            dim=-1)
        X = torch.cat((X, phi), dim=-1)

        alpha = torch.unsqueeze(torch.arctan(torch.div(X[:, :, 2], X[:, :, 1])), dim=-1)
        X = torch.cat((X, alpha), dim=-1)

        beta = torch.unsqueeze(torch.arctan(torch.div(X[:, :, 2], X[:, :, 0])), dim=-1)
        X = torch.cat((X, beta), dim=-1)

    if add_freq_domain_features:
        fft_frontal = fft(X.numpy()[:, :, 0])
        fft_vertical = fft(X.numpy()[:, :, 1])
        fft_lateral = fft(X.numpy()[:, :, 2])

        for fft_axis in [fft_lateral, fft_vertical, fft_frontal]:
            for method in [lambda x, axis: np.sum(np.square(x), axis=axis), np.max, np.min, np.mean]:
                feature = torch.unsqueeze(torch.tensor(np.real(method(fft_axis, 1))).repeat(X.shape[1], 1).T, dim=-1)
                X = torch.cat((X, feature), dim=-1)

    return X
