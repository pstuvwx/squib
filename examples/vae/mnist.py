import torch
import torch.nn    as nn
import torch.optim as optim
import torchvision

from squib.updaters.updater import StanderdUpdater
from squib.trainer.trainer  import Trainer



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 40),
        )
    

    def forward(self, x):
        x = x.reshape(-1, 784)

        x = self.mlp(x)
        mean, log_std = torch.chunk(x, chunks=2, dim=1)

        return mean, log_std



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    
    def forward(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        x = mean + std*eps

        x = self.mlp(x)
        x = x.reshape(-1, 1, 28, 28)

        return x



def VAEUpdater(encoder, decoder, optimizer=None, tag=None) -> StanderdUpdater:
    bce = nn.BCELoss()
    kld = lambda m, l: 0.5 * torch.mean(1 + 0.5*l - m.pow(2) - l.exp().pow(2))

    def _loss_func(x, _):
        mean, log_std = encoder(x)
        y = decoder(mean, log_std)

        loss_bce = bce(y, x.detach())
        loss_kld = kld(mean, log_std)
        loss     = loss_bce - loss_kld

        result = {
            'bce':loss_bce.item(),
            'kld':loss_kld.item(),
        }
        return loss, result

    upd = StanderdUpdater(loss_func=_loss_func,
                          optimizer=optimizer,
                          tag      =tag)
    
    return upd



def main():
    trainset = torchvision.datasets.MNIST(root='./mnist', train=True,  download=True,
                                          transform=torchvision.transforms.ToTensor())
    testset  = torchvision.datasets.MNIST(root='./mnist', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

    train_loader      = torch.utils.data.DataLoader(trainset,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=2)
    validation_loader = torch.utils.data.DataLoader(testset,
                                                    batch_size=128,
                                                    shuffle=False,
                                                    num_workers=2)

    device = torch.device('cuda:0')

    enc = Encoder()
    dec = Decoder()
    opt = optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-3)
    enc.to(device)
    dec.to(device)

    train_updater      = VAEUpdater(enc, dec, tag='tr', optimizer=opt)
    validation_updater = VAEUpdater(enc, dec, tag='vl')

    trainer = Trainer(loader =train_loader,
                      updater=train_updater,
                      device =device,
                      save_to='./result')

    trainer.log_report(keys   =['tr/bce', 'vl/bce', 'tr/kld', 'vl/kld'],
                       trigger=(1, 'epoch'))
    
    trainer.add_evaluation(loader =validation_loader,
                           updater=validation_updater,
                           trigger=(1, 'epoch'))
    
    trainer.save_model(path   ='models/encoder_{epoch}.pth',
                       model  =enc,
                       trigger=(1, 'epoch'))
    trainer.save_model(path   ='models/decoder_{epoch}.pth',
                       model  =dec,
                       trigger=(1, 'epoch'))
                       
    trainer.save_trainer(path   ='trainer.pth',
                         models ={'encoder':enc, 'decoder':dec, 'opt':opt},
                         trigger=(1, 'epoch'))
    
    trainer.run()


if __name__ == "__main__":
    main()
    