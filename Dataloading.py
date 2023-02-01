from glob import glob
import os
import pandas as pd
import numpy as np
from scipy import interpolate


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

            interpolated_df['file_name'] = record.split('/')[-1]
            interpolated_df['frame_id'] = interpolated_df.index
            patient_data = pd.concat((patient_data, interpolated_df), axis=0)
        else:
            record_data = pd.read_csv(record, header=None)
            record_data['file_name'] = record.split('/')[-1]
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

            interpolated_df['file_name'] = record.split('/')[-1]
            interpolated_df['frame_id'] = interpolated_df.index
            patient_data = pd.concat((patient_data, interpolated_df), axis=0)
        else:
            record_data = pd.read_csv(record, header=None)
            record_data['file_name'] = record.split('/')[-1]
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
