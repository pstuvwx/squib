import torch
import torch.nn    as nn
import torch.optim as optim
import torchvision

from updaters.supervised import ClassificationUpdater

def MLP():
    class Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)

    mlp = nn.Sequential(
        nn.Conv2d(  1, 128, kernel_size=3, stride=1, padding=1),    # (-1, 128, 28, 28)
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),    # (-1, 128, 14, 14)
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),    # (-1, 128,  7,  7)
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),    # (-1, 128,  3,  3)
        nn.ReLU(inplace=True),
        Flatten(),                                                  # (-1, 128*3*3)
        nn.Linear(128*3*3, 10)
    )

    return mlp



def main():
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True,  download=True)
    testset  = torchvision.datasets.MNIST(root='./mnist', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

    model = MLP()
    opt   = optim.Adam(model.parameters())
    
