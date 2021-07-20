import torch
import torchvision
import time
import json
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

from models.epoch import Epoch
from models.networkRun import NetworkRun

class RunManager():
    def __init__(self):
        self.epoch = Epoch()
        self.run = NetworkRun()
        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run.start_time = time.time()
        self.run.params = run

        self.network = network
        self.loader = loader

        self.tb = SummaryWriter(comment=f'-{run}')
        
        images, _ = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(
            self.network,
            images.to(getattr(run, 'device', 'cpu'))
        )

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