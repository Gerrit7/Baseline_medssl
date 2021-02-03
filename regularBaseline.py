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
from medssl.data.medmnist.info import INFO

flag = 'octmnist'
baseline_type = 'regular'

dataset_path = os.path.join('Baseline_' + flag, baseline_type)
model_save_path = os.path.join(dataset_path, 'Checkpoints')
csv_save_path = os.path.join(dataset_path, 'Evaluation')
train_batch_size = 128
valid_batch_size= 128

info_dict = {
        'Type': 'Baseline without Dropout',
        'Dataset': 'OCTMMNIST',
        'Model': 'BaselineResnet',
        'Optimizer': 'SGD',
        'Criterion': 'CrossEntropyLoss',
        'Sheduler': 'ReduceLROnPlateau',
}

######################################################################################################
######################################################################################################
##read in data (default is pathmnist)
traindata = MedMNIST(root='../medssl/data/medmnist', split='train', transform=transforms.ToTensor(), download=True, flag=flag)
validdata = MedMNIST(root='../medssl/data/medmnist', split='val', transform=transforms.ToTensor(), download=True, flag=flag)

sampler = BalancedSampler(traindata, n=7754, shuffle=True) #Notiz: aktuell kein Seed gesetzt. Wird jedes Mal neue Indices auswählen
trainloader = DataLoader(dataset=traindata, batch_size=train_batch_size, sampler=sampler, drop_last=True, num_workers=4)
validloader = DataLoader(dataset=validdata, batch_size=valid_batch_size, num_workers=4)


######################################################################################################
######################################################################################################
# check for gpu
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

######################################################################################################
######################################################################################################
## model setup
info = INFO[flag]
in_channels = info['n_channels']
n_classes = len(info['label'])

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


def csv_results(fieldnames_dict, csv_path):
    # save results in csv file
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    file_exists = os.path.isfile(os.path.join(csv_save_path, 'results.csv'))
    with open(os.path.join(csv_save_path, 'results.csv'), 'a+', newline='') as csvfile:
        fieldnames = [*fieldnames_dict]

        infowriter = csv.DictWriter(csvfile, fieldnames=[*info_dict])
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            infowriter.writeheader()
            infowriter.writerow(info_dict)
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerow(fieldnames_dict)


######################################################################################################
######################################################################################################
## training procedure

epochs = 1000

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

best_loss = 1000
best_epoch = 0
losses = np.ndarray((epochs,2))

for it in trange(epochs):

    train_loss = train()

    valid_loss = validate()
    scheduler.step(valid_loss)  # learning rate is reduced if validation loss does not improve anymore

    #save 'best model' if model is better than previous epochs
    if valid_loss < best_loss:
        torch.save(model.state_dict(), os.path.join(model_save_path, "bestmodel.pth")) #no training parameters are saved
        best_loss = valid_loss
        best_epoch = it


    losses[it,0] = train_loss
    losses[it,1] = valid_loss
    #plotLosses(it, losses)  # saves plot in file

    result_dict = {
        'Batch_Size Training': train_batch_size,
        'Batch_Size Validation': valid_batch_size,
        'Epoche': it,
        'Train Loss': train_loss,
        'Valid Loss': valid_loss,
    }
    csv_results(fieldnames_dict=result_dict, csv_path=csv_save_path)