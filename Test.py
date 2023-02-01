from matplotlib import pyplot as plt
from Model import MyRNN
from Dataloading import get_patient_data

X, y = get_patient_data(use_interpolation=True)

model = MyRNN(in_features=7, hidden_features=10, num_layers=2)


