from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from resources.plotcm import plot_confusion_matrix

from itertools import product

from torch.utils.tensorboard import SummaryWriter

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([]).cuda()
    for batch in loader:
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        ).cuda()
    return all_preds

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

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

parameters = dict(
    lr = [.01, .001]
    ,batch_size = [10, 100, 1000]
)
param_values = [v for v in parameters.values()]

for lr, batch_size in product(*param_values):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    comment = f' batch_size={batch_size} lr={lr}'
    tb = SummaryWriter(comment=comment)

    network = Network()
    network = network.cuda()

    images, labels = next(iter(train_loader))
    images, labels = images.cuda(), labels.cuda()
    grid = torchvision.utils.make_grid(images)

    tb.add_image('images', grid)
    tb.add_graph(network, images)
    tb.close()

    optimizer = optim.Adam(network.parameters(), lr=lr)

    for epoch in range(10):
        total_loss = 0
        total_correct = 0

        for batch in train_loader:
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()

            preds = network(images) # feed batch forward through the network
            loss = F.cross_entropy(preds, labels) # calculate the loss

            optimizer.zero_grad() # zero out the gradients
            loss.backward() # calculate the gradients
            optimizer.step() # update the weights

            total_loss += loss.item() * batch_size
            total_correct += get_num_correct(preds, labels)

        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        print("batch_size:", batch_size, "lr:", lr, "epoch:", epoch, "total_correct:", total_correct, "loss:", total_loss)

    train_preds = get_all_preds(network, train_loader)
    preds_correct = get_num_correct(train_preds.cuda(), train_set.targets.cuda())

    print("total correct:", preds_correct)
    print("accuracy:", preds_correct / len(train_set))

    stacked = torch.stack(
        (
            train_set.targets.cuda()
            ,train_preds.argmax(dim=1).cuda()
        )
        ,dim=1
    )

    cmt = torch.zeros(10, 10, dtype=torch.int32)

    for p in stacked.cuda():
        tl, pl = p.tolist()
        cmt[tl, pl] = cmt[tl, pl] + 1

    print(cmt.cuda())

# cm = confusion_matrix(train_set.targets.cpu(), train_preds.argmax(dim=1).cpu())
# names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
# plt.figure(figsize=(10,10))
# plot_confusion_matrix(cm, names)