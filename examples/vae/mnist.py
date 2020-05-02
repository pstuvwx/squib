import os

import numpy       as np
import torch
import torch.nn    as nn
import torch.optim as optim
from PIL                    import Image
from torch.utils.data       import DataLoader, Dataset
from torchvision.datasets   import MNIST
from torchvision.transforms import ToTensor

from squib.updaters.updater import StanderdUpdater
from squib.trainer.trainer  import Trainer


latent_dim = 20

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, latent_dim*2),
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
            nn.Linear(latent_dim, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )

    
    def forward(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        x   = mean + std*eps

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



def example(encoder       :Encoder,
            decoder       :Decoder,
            validation_set:Dataset,
            save_to       :str,
            device        :torch.device):
    n_img           = 8
    validation_imgs = torch.stack([validation_set[i][0] for i in range(n_img)])

    if not os.path.exists(save_to):
        os.mkdir(save_to)

    def _func():
        with torch.no_grad():
            input_img = validation_imgs.to(device)
            mean      = torch.zeros((8, latent_dim), dtype=torch.float32, device=device)
            std       = torch.ones ((8, latent_dim), dtype=torch.float32, device=device)

            reconstructed = decoder(*encoder(input_img))*255
            generated     = decoder(mean, std)*255

            reconstructed = reconstructed.detach().cpu().numpy()
            generated     = generated    .detach().cpu().numpy()
        
        name = ['_reconstructed.png', '_generated.png']
        for j, rg in enumerate(zip(reconstructed, generated)):
            for n, i in zip(name, rg):
                path = os.path.join(save_to, str(j)+n)
                img  = i.reshape(28, 28).astype(np.uint8)
                img  = Image.fromarray(img)
                img.save(path)

    return _func



def main():
    trainset      = MNIST(root     ='./mnist',
                          train    =True,
                          download =True,
                          transform=ToTensor())
    validationset = MNIST(root     ='./mnist',
                          train    =False,
                          download =True,
                          transform=ToTensor())

    train_loader      = DataLoader(trainset,
                                   batch_size =128,
                                   shuffle    =True,
                                   num_workers=2)
    validation_loader = DataLoader(validationset,
                                   batch_size =128,
                                   shuffle    =False,
                                   num_workers=2)

    device = torch.device('cuda:0')

    enc = Encoder()
    dec = Decoder()
    opt = optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=1e-4)
    enc.to(device)
    dec.to(device)

    train_updater      = VAEUpdater(enc, dec, tag='tr', optimizer=opt)
    validation_updater = VAEUpdater(enc, dec, tag='vl')

    trainer = Trainer(loader =train_loader,
                      updater=train_updater,
                      device =device,
                      save_to='./result')

    trainer.log_report(keys   =['tr/bce', 'vl/bce', 'tr/kld', 'vl/kld'],
                       plots  ={
                           'bce.png':['tr/bce', 'vl/bce'],
                           'kld.png':['tr/kld', 'vl/kld']
                           },
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

    trainer.add_event(example(enc, dec, validationset, './example', device),
                      trigger=(1, 'epoch'))
    
    trainer.run()


if __name__ == "__main__":
    main()
    