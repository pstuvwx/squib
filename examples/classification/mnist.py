import torch
import torch.nn    as nn
import torch.optim as optim

from torch.utils.data       import DataLoader
from torchvision.datasets   import MNIST
from torchvision.transforms import ToTensor

from squib.functions.evaluation import accuracy
from squib.updaters.updater     import StanderdUpdater
from squib.trainer.trainer      import Trainer


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



def ClassificationUpdater(model, optimizer=None, tag=None) -> StanderdUpdater:
    cel = nn.CrossEntropyLoss()

    def _loss_func(x, t):
        y = model(x)
        loss = cel(y, t)
        accu = accuracy(y, t)
        result = {
            'loss':loss.item(),
            'accuracy':accu
        }
        return loss, result

    upd = StanderdUpdater(loss_func=_loss_func,
                          optimizer=optimizer,
                          tag      =tag)
    
    return upd


def main():
    trainset      = MNIST(root='./mnist', train=True,  download=True,
                          transform=ToTensor())
    validationset = MNIST(root='./mnist', train=False, download=True,
                          transform=ToTensor())

    train_loader      = DataLoader(trainset,
                                   batch_size=128,
                                   shuffle=True,
                                   num_workers=2)
    validation_loader = DataLoader(validationset,
                                   batch_size=128,
                                   shuffle=False,
                                   num_workers=2)

    model  = MLP()
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda:0')
    model.to(device)

    train_updater      = ClassificationUpdater(model, tag='tr', optimizer=opt)
    validation_updater = ClassificationUpdater(model, tag='vl')

    trainer = Trainer(loader =train_loader,
                      updater=train_updater,
                      device =device,
                      save_to='./result')

    trainer.log_report(keys   =['tr/loss', 'vl/loss', 'tr/accuracy', 'vl/accuracy'],
                       plots  ={
                           'loss.png'    :['tr/loss', 'vl/loss'],
                           'accuracy.png':['tr/accuracy', 'vl/accuracy'],
                       },
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
    