# Non Data Augmentation
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from Load_data import SR_dataset
from model import Generator, Discriminator
from Config import configure
from train import train
"""
    DATA LOADER
"""
batch_size= configure.batch_size

hr_path = configure.hr_path
lr_path = configure.lr_path
transform = transforms.Compose([transforms.ToTensor()])
# Input은 [0,1]이고 Output은 [-1,1]이다.
train_dataset = SR_dataset(lr_path = lr_path, hr_path = hr_path,transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 4, shuffle =True)


"""
    TRAIN
"""

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)
optG = torch.optim.Adam(netG.parameters(),lr=1e-4)
optD = torch.optim.Adam(netD.parameters(),lr=1e-4)

Nets = [netG, netD]
Optimizers = [optG, optD]

train(Nets, Optimizers, train_loader, train_dataset, DEVICE)

"""
    JUST SHOW
"""

netG.eval()

k = train_dataset[5][0].to(DEVICE)
pred =netG(k.view(1,3,96,96))

import matplotlib.pyplot as plt

pred = pred.cpu().detach().numpy()
pred = pred.transpose(0,2,3,1)
pred = pred/2 + 0.5

pred = pred.reshape((384,384,3))
plt.imshow(pred)
