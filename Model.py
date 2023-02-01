import torch


class MyRNN(torch.nn.Module):

    def __init__(self, in_features, hidden_features, num_layers, use_dropout=0):
        super().__init__()
        self.rnn_layer = torch.nn.RNN(input_size=in_features, hidden_size=hidden_features, num_layers=num_layers,
                                      batch_first=True, dropout=use_dropout)

        self.fc = torch.nn.Linear(in_features=hidden_features, out_features=4)

    def forward(self, x):

        output = self.rnn_layer(x)

        output = self.fc(output)

        return output