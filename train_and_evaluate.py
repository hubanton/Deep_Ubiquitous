import io
import os
import numpy as np
import torch
import sklearn.metrics as sk
import yaml
from tqdm import tqdm
from Dataloading import get_dataloaders
from Model import MyRNN


def train_step(module, criterion, optimizer, x, y, scheduler, device, scaler):
    module.train()

    # zero the gradients to not accumulate every gradient calculated
    optimizer.zero_grad()

    # use autocast for mixed precision training
    with torch.autocast(device_type=device):
        output = module(x)
        loss = criterion(output, y)

    # modified backward and step to work with scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # update scheduler
    if scheduler is not None:
        scheduler.step()

    return loss


def train_and_evaluate(module, criterion, optimizer, train_loader, epochs, validation_loader, savefile_name,
                       device, scaler, use_last_module=False, scheduler=None):
    # if we want to continue training check if last trained module exists and the load it ( for 1 g) )
    if use_last_module:
        if os.path.exists(savefile_name + '.pth'):
            last_checkpoint = torch.load(savefile_name + '.pth')
            if last_checkpoint is not None:
                module.load_state_dict(last_checkpoint['model'])
                optimizer.load_state_dict(last_checkpoint['optimizer'])

    # list to keep track of performance on validation set in every epoch
    epoch_performance = {"accuracy": [], "bal_accuracy": [], "macro": [], "weighted": []}

    pbar = tqdm(range(epochs))

    for epoch in pbar:

        loss_list = []

        for k, (batch_sequences, batch_labels) in enumerate(train_loader):
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            module.train()
            loss = train_step(module, criterion, optimizer, batch_sequences, batch_labels, scheduler, device, scaler)

            loss_list.append(loss)

        loss_list = torch.tensor(loss_list)

        module.eval()

        pred = torch.tensor([]).to(device)
        labels = torch.tensor([])

        with torch.no_grad():

            # calculate predictions batch wise to not overload GPU
            for k, (val_sequences, val_labels) in enumerate(validation_loader):
                val_sequences = val_sequences.to(device)

                # get prediction of model on validation batch
                batch_pred = torch.argmax(module(val_sequences), dim=-1).to(torch.int)

                pred = torch.cat([pred, batch_pred], dim=0)
                labels = torch.cat([labels, val_labels], dim=0)

        pred = pred.to('cpu').detach().numpy().astype(np.int)
        one_hot_pred = np.zeros((pred.size, 4), dtype=np.int)
        one_hot_pred[np.arange(pred.size), pred] = 1

        labels = labels.to('cpu').detach().numpy().astype(np.int)
        one_hot_labels = np.zeros((labels.size, 4), dtype=np.int)
        one_hot_labels[np.arange(labels.size), labels] = 1

        accuracy = sk.accuracy_score(one_hot_labels, one_hot_pred)
        balanced_accuracy = sk.balanced_accuracy_score(labels, pred)
        macro_performance = sk.f1_score(one_hot_labels, one_hot_pred, average="macro")
        weighted_performance = sk.f1_score(one_hot_labels, one_hot_pred, average="weighted")

        # check if we got a better performance and save the weights and optimizer states if that is the case
        # if len(epoch_performance) == 0 or macro_performance > max(epoch_performance):
        #     checkpoint = {
        #         'model': module.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     }
        #     torch.save(checkpoint, savefile_name + ' (best)' + '.pth')

        # save current trained model in case of interruption ( for 1 g) )
        checkpoint = {
            'model': module.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, savefile_name + ' (last)' + '.pth')

        epoch_performance["accuracy"].append(accuracy)
        epoch_performance["bal_accuracy"].append(balanced_accuracy)
        epoch_performance["macro"].append(macro_performance)
        epoch_performance["weighted"].append(weighted_performance)

        pbar.set_postfix({key: value[-1] for key, value in epoch_performance.items()})

    with open(f"Results_({savefile_name}).txt", "w") as text_file:
        text_file.write(str(epoch_performance))

    return epoch_performance


def rnn_training(X, y, savefile_name, device, backbone, in_features=7, hidden_features=10, num_layers=2, epochs=100):
    train_dataloader, val_dataloader = get_dataloaders(X, y)

    model = MyRNN(in_features, hidden_features, num_layers, backbone).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()

    grad_scaler = torch.cuda.amp.GradScaler()

    performance = train_and_evaluate(model, criterion, optimizer, train_dataloader, epochs, val_dataloader,
                                     savefile_name, device, grad_scaler, use_last_module=False, scheduler=None)

    return performance
