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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from medssl.datasets.medmnist import MedMNIST, BalancedSampler
from medssl.models.resnet import BayesianResNet18_v2
from medssl.data.medmnist.info import INFO

flag = 'octmnist'
baseline_type = 'dropout_conv'

dataset_path = os.path.join('Baseline_' + flag, baseline_type)
model_path = os.path.join(dataset_path, 'Checkpoints', 'bestmodel.pth')

csv_save_path = os.path.join(dataset_path, 'Testing')
test_batch_size= 1

info_dict = {
        'Training': 'Without dropout',
        'Testing': 'With MC Dropout',
        'Dataset': 'OCTMMNIST',
        'Model': 'BaselineResnet',
}

######################################################################################################
######################################################################################################
##read in data (default is pathmnist)
testdata = MedMNIST(root='../medssl/data/medmnist', split='test', transform=transforms.ToTensor(), download=True, flag=flag)
testloader = DataLoader(dataset=testdata, batch_size=test_batch_size, num_workers=4)


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

model = BayesianResNet18_v2(num_classes=n_classes, in_channels=in_channels, mnist=True).to(device)  #important to move model to gpu before creating optimizer
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

######################################################################################################
######################################################################################################
## define evaluation routines

def test(mc=False, t_in=0):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):

            m = nn.Softmax(dim=1).to(device)  # softmax computes class probabilites from logits, does not change order
            mean_variance = -1
            confidence = -1

            if mc:
                output_list = []
                # getting outputs for T forward passes (Monte Carlo Algorithm)
                for i in range(t_in):
                    m = nn.Softmax(dim=1)
                    output_list.append(torch.unsqueeze(m(model(inputs)), 0))

                # calculating mean/variance
                output_mean = torch.cat(output_list, 0).mean(0)
                mean_variance = torch.cat(output_list, 0).var(0).mean().item()
                confidence = output_mean.data.cpu().numpy().max()
                pred = output_mean.data.cpu().numpy().argmax()
                prob = m(outputs)

            else:
                outputs = model(inputs.to(device))
                pred = torch.argmax(outputs, dim=1)  # before softmax
                prob = m(outputs)

            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            #y_true.extend(targets.squeeze().cpu().numpy())  # Option 1 to obtain all true labels
            y_true.extend(targets[0].cpu().numpy())
            result_dict = {
                'Batch Idx': batch_idx,
                'Y_pred': pred.cpu().numpy(),
                'Y_prob': prob.cpu().numpy(),
                'Y_true': targets.squeeze().cpu().numpy(),
                'variance': mean_variance,
                'confidence': confidence
            }
            csv_results(fieldnames_dict=result_dict, csv_path=csv_save_path)


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
## testing procedure

y_pred = []
y_prob = []
y_true = []

test_loss = test(mc=True, t_in=10)

print(len(y_true))
print(len(np.asarray(y_prob)))
auc = roc_auc_score(y_true, np.asarray(y_prob), multi_class="ovo")
acc = accuracy_score(y_true, y_pred)

conf = confusion_matrix(y_true, y_pred, normalize='all')
print(conf)
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot()
plt.show()


histogram = np.bincount(y_true)/len(y_true) # true class distribution
print(histogram)
plt.bar(range(len(histogram)), histogram)
plt.show()

