import pandas as pd
import numpy as np
import torch
from dataset import SP500Dataset
from model import Generator, Discriminator
from train import train
from test import plot_generated_paths
from torch.utils.data import DataLoader
import yfinance as yf
import random

# Data processing and loading
sp500 = yf.download('^GSPC', '2009-05-01', '2018-12-31')
sp500log = np.log(sp500['Close'] / sp500['Close'].shift(1))[1:].values
# (Process sp500log, calculate params, and normalize as in original code)

# Model and Training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG = Generator(nz, 1).to(device)
netD = Discriminator(1, 1).to(device)

dataset = SP500Dataset(sp500processed, 127)
dataloader = DataLoader(dataset, batch_size=30, shuffle=True)
train(netG, netD, dataloader, device)

# Test and Plot Results
plot_generated_paths(netG, device, sp500max, params, sp500log_mean)
