import os
os.chdir(os.path.abspath(__file__ + "/../"))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import trange

from medssl.datasets.medmnist import MedMNIST, BalancedSampler
from medssl.models.resnet import BaselineResNet

######################################################################################################
######################################################################################################
##read in data (default is pathmnist)
traindata = MedMNIST(root='../medssl/data/medmnist', split='train', transform=transforms.ToTensor(), download=True, flag='octmnist')
validdata = MedMNIST(root='../medssl/data/medmnist', split='val', transform=transforms.ToTensor(), download=True, flag='octmnist')

sampler = BalancedSampler(traindata, n=2000, shuffle=True) #Notiz: aktuell kein Seed gesetzt. Wird jedes Mal neue Indices auswählen
trainloader = DataLoader(dataset=traindata, batch_size=128, sampler=sampler, drop_last=True, num_workers=4)
validloader = DataLoader(dataset=validdata, batch_size=1, num_workers=4)


######################################################################################################
######################################################################################################
# check for gpu
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

######################################################################################################
######################################################################################################
## model setup

in_channels = 3     # params for pathmnist
n_classes = 9     # params for pathmnist

model = BaselineResNet(num_classes=n_classes, in_channels=in_channels, mnist=True).to(device)  #important to move model to gpu before creating optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

######################################################################################################
######################################################################################################
## define training and validation routine

def train():
    model.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device)
        targets = targets.squeeze().to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        batch_loss = criterion(outputs, targets)
        train_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()

    return train_loss/batch_idx


def validate():
    model.eval()  # sets dropout and batch normalization layers to evaluation mode
    valid_loss = 0

    with torch.no_grad(): #turns off gradient computation
        for batch_idx, (inputs, targets) in enumerate(validloader):

            inputs = inputs.to(device)
            targets = targets.squeeze(dim=1).to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            valid_loss += batch_loss.item()

        return valid_loss/batch_idx


def csv_results(fieldnames_dict, csv_path="mc_evaluation"):
    # save results in csv file
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    file_exists = os.path.isfile(os.path.join(csv_path, 'baseline_results.csv'))
    with open(os.path.join(csv_path, 'baseline_results.csv'), 'a+', newline='') as csvfile:
        fieldnames = fieldnames_dict.keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(fieldnames)


######################################################################################################
######################################################################################################
## training procedure

epochs = 2

modelpath_best = 'checkpoints/bestmodel.pth'
best_loss = 1000
best_epoch = 0
losses = np.ndarray((epochs,2))

for it in trange(epochs):

    train_loss = train()
    print(train_loss)

    valid_loss = validate()
    scheduler.step(valid_loss)  # learning rate is reduced if validation loss does not improve anymore

    #save 'best model' if model is better than previous epochs
    if valid_loss < best_loss:
        torch.save(model.state_dict(), modelpath_best) #no training parameters are saved
        best_loss = valid_loss
        best_epoch = it


    losses[it,0] = train_loss
    losses[it,1] = valid_loss
    #plotLosses(it, losses)  # saves plot in file

    dict = {
        'Dataset': 'OCTMMNIST',
        'Model': 'BaselineResnet',
        'Optimizer': 'SGD',
        'Criterion': 'CrossEntropyLoss',
        'Sheduler': 'ReduceLROnPlateau',
        'Epoche': it,
        'Train Loss': train_loss,
        'Valid Loss': valid_loss,
    }
    csv_results(fieldnames_dict=dict, csv_path='Baseline_OCTMNIST')