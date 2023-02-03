from Dataloading import create_windows, get_patient_data

window_sizes = [200, 200, 400, 400, 800, 800]
hop_sizes = [20, 40, 40, 80, 80, 160]

X, y = get_patient_data(use_interpolation=True)

for win_size, hop_size in zip(window_sizes, hop_sizes):

    create_windows(X, y, win_size, hop_size)
