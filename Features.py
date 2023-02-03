from Dataloading import create_windows, get_patient_data

window_sizes = [400, 1200, 2000, 4000]
hop_sizes = [80, 240, 400, 800]

X, y = get_patient_data(use_interpolation=True)

for win_size, hop_size in zip(window_sizes, hop_sizes):

    create_windows(X, y, win_size, hop_size)
