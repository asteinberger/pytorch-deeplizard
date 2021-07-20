import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import datetime
import pandas as pd

from collections import OrderedDict

from models.runManager import RunManager
from models.runBuilder import RunBuilder
from models.network import Network

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
pd.set_option('display.width', 120)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))
data = next(iter(loader))
mean = data[0].mean()
std = data[0].std()

train_set_normal = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
        ,transforms.Normalize(mean, std)
    ])
)

trainsets = {
    'not_normal': train_set
    ,'normal': train_set_normal
}

params = OrderedDict(
    lr = [.01]
    ,batch_size = [1000]
    ,num_epochs = [20]
    ,device = ['cuda']
    ,trainset = ['not_normal', 'normal']
)

m = RunManager()
for run in RunBuilder.get_runs(params):

    device = torch.device(run.device)
    network = Network().to(device)
    loader = torch.utils.data.DataLoader(trainsets[run.trainset], batch_size=run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(run.num_epochs):
        m.begin_epoch()
        for batch in loader:

            images = batch[0].to(device)
            labels = batch[1].to(device)
            preds = network(images) # feed batch forward through the network
            loss = F.cross_entropy(preds, labels) # calculate the loss
            optimizer.zero_grad() # zero out the gradients
            loss.backward() # calculate the gradients
            optimizer.step() # update the weights

            m.track_loss(loss)
            m.track_num_correct(preds, labels)

        m.end_epoch()
    m.end_run()
now = datetime.datetime.now().__str__().translate ({ord(c): "-" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+ "})
m.save(f'results-{now}')