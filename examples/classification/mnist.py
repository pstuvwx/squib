import torch
import torch.nn    as nn
import torch.optim as optim
import torchvision

from updaters.supervised import ClassificationUpdater
from trainer.trainer     import Trainer


def MLP():
    class Flatten(nn.Module):
        def forward(self, x):
            return torch.flatten(x, 1)

    mlp = nn.Sequential(
        nn.Conv2d( 1, 32, kernel_size=4, stride=2, padding=1),    # (-1, 128, 14, 14)
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),    # (-1, 128,  7,  7)
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),    # (-1, 128,  3,  3)
        nn.ReLU(inplace=True),
        Flatten(),                                                # (-1, 128*3*3)
        nn.Linear(32*3*3, 10)
    )

    return mlp



def main():
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True,  download=True,
                                          transform=torchvision.transforms.ToTensor())
    testset  = torchvision.datasets.MNIST(root='./mnist', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

    train_loader      = torch.utils.data.DataLoader(trainset,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    num_workers=2)
    validation_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=64,
                                                    shuffle=False,
                                                    num_workers=2)

    model  = MLP()
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cpu')

    train_updater      = ClassificationUpdater(model, tag='tr', optimizer=opt)
    validation_updater = ClassificationUpdater(model, tag='vl')

    trainer = Trainer(loader =train_loader,
                      updater=train_updater,
                      device =device,
                      save_to='./result')

    trainer.log_report(keys   =['tr/loss', 'vl/loss', 'tr/accuracy', 'vl/accuracy'],
                       trigger=(1, 'epoch'))
    
    trainer.add_evaluation(loader =validation_loader,
                           updater=validation_updater,
                           trigger=(1, 'epoch'))
    
    trainer.save_model(path   ='models/models_{epoch}.pth',
                       model  =model,
                       trigger=(1, 'epoch'))
    trainer.save_trainer(path   ='trainer.pth',
                         models ={'model':model, 'opt':opt},
                         trigger=(1, 'epoch'))
    
    trainer.run()


if __name__ == "__main__":
    main()
    