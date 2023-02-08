import ast

import matplotlib.pyplot as plt
import pandas as pd

metrics = ['accuracy', 'weighted', 'bal_accuracy', 'macro']


def attribute_results_to_csv():
    results = []

    for metric in metrics:
        features = []
        for use_acc_features in [True, False]:
            for use_freq_domain_features in [True, False]:
                with open(f'results/Results_(acc_f {use_acc_features}, freq_f {use_freq_domain_features}).txt') as f:
                    data = f.read()
                    js = ast.literal_eval(data)
                    res = round(max(js[metric]), 4)
                    features.append(res)

        results.append(features)
    df = pd.DataFrame(results, columns=['Used Attributes:' 'Acc + Freq', 'Frequency', 'Acc', 'Default'], index=metrics)
    df.to_csv('attributes.csv')


def window_size_results_to_csv():
    results = []
    window_size = [200, 200, 400, 400, 800, 800]
    hop_sizes = [20, 40, 40, 80, 80, 160]
    for metric in metrics:
        features = []
        for window_size, hop_size in zip(window_size, hop_sizes):
            with open(f'results/Results_(Window size {window_size}, Hop size {hop_size}).txt') as f:
                data = f.read()
                js = ast.literal_eval(data)
                res = round(max(js[metric]), 3)
                features.append(res)

        results.append(features)
    columns = [f"{window_size, hop_size}" for window_size, hop_size in zip(window_size, hop_sizes)]
    df = pd.DataFrame(results, columns=columns, index=metrics)
    df.to_csv('window_sizes.csv')


def plot_model_results():
    models = ['GRU', 'LSTM', 'RNN']

    for metric in metrics:
        for model in models:
            with open(f'results/Results_({model}).txt') as f:
                data = f.read()
                js = ast.literal_eval(data)
                res = js[metric]
                plt.plot(range(len(res)), res, label=f'{model}')
        plt.legend()
        ax = plt.gca()
        ax.set_ylim([0.6, 1])
        plt.savefig(f"{metric}.png", bbox_inches='tight', pad_inches=0)
        plt.clf()


attribute_results_to_csv()
window_size_results_to_csv()
plot_model_results()
