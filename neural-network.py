from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict
from itertools import product

import time
import datetime
import pandas as pd
import json

from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
pd.set_option('display.width', 120)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = F.relu(self.fc1(t.flatten(start_dim=1)))
        t = F.relu(self.fc2(t))
        t = self.out(t)

        return t

class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class Epoch():
    def __init__(self):
        self.count = 0
        self.loss = 0
        self.num_correct = 0
        self.start_time = None

class Run():
    def __init__(self):
        self.params = None
        self.count = 0
        self.data = []
        self.start_time = None

class RunManager():
    def __init__(self):
        self.epoch = Epoch()
        self.run = Run()
        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run.start_time = time.time()
        self.run.params = run

        self.network = network
        self.loader = loader

        self.tb = SummaryWriter(comment=f'-{run}')

        images, labels = next(iter(self.loader))
        images, labels = images.cuda(), labels.cuda()

        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)

    def end_run(self):
        self.tb.close()
        self.epoch.count = 0

    def begin_epoch(self):
        self.epoch.start_time = time.time()
        self.epoch.count += 1
        self.epoch.loss = 0
        self.epoch.num_correct = 0

    def end_epoch(self):
        epoch_duraction = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        loss = self.epoch.loss / len(self.loader.dataset)
        accuracy = self.epoch.num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch.count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch.count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duraction
        results["run duration"] = run_duration
        for k, v in self.run.params._asdict().items(): results[k] = v
        
        self.run.data.append(results)
        df = pd.DataFrame.from_dict(self.run.data, orient='columns')
        print(df)

    def track_loss(self, loss):
        self.epoch.loss += loss.item() * self.loader.batch_size
    
    def track_num_correct(self, preds, loaders):
        self.epoch.num_correct += self._get_num_correct(preds, loaders)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run.data
            ,orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [10, 100, 1000]
    ,num_epochs = [5, 10]
)

m = RunManager()
for run in RunBuilder.get_runs(params):

    network = Network()
    network = network.cuda()
    loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(run.num_epochs):
        m.begin_epoch()
        for batch in loader:

            images, labels = batch
            images, labels = images.cuda(), labels.cuda()
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