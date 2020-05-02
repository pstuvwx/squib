import os

import numpy       as np
import torch
import torch.nn    as nn
import torch.optim as optim
from PIL                    import Image
from torch.utils.data       import DataLoader, Dataset
from torchvision.datasets   import MNIST
from torchvision.transforms import ToTensor

from squib.updaters.updater import MultilossUpdater
from squib.trainer.trainer  import Trainer



def pixelwise_std(x):
    b, _, h, w = map(int, x.shape)

    mx  = torch.mean(x,  dim=0, keepdim=True)
    
    var = torch.mean((x-mx)**2, dim=0, keepdim=True)
    std = (var+1e-8) ** 0.5
    
    dst = torch.mean(std, dim=1, keepdim=True)
    dst = dst.expand(b, 1, h, w)
    
    return dst



class LambdaBlock(nn.Module):
    def __init__(self, func):
        super(LambdaBlock, self).__init__()

        self.func = func


    def forward(self, h):
        h = self.func(h)
        return h



def Generater():
    net = nn.Sequential(
        nn.Linear(128, 64*7*7),
        nn.ReLU(),
        LambdaBlock(lambda x: x.reshape(-1, 64, 7, 7)),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32,  1, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )
    return net



def Discriminater():
    net = nn.Sequential(
        nn.Conv2d( 1, 32, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(),
        LambdaBlock(lambda x: torch.cat([x, pixelwise_std(x)], dim=1)),
        LambdaBlock(lambda x: torch.flatten(x, 1)),
        nn.Linear((64+1)*7*7, 1),
    )
    return net

    

def GANUpdater(gene, disc, optimizers=None, tag=None, *arg, **karg) -> MultilossUpdater:
    def _loss_func(x, _):
        b = int(x.shape[0])
        z = torch.randn((b, 128), dtype=torch.float32, device=x.device)
        y = gene(z)

        d_y = disc(y)
        d_z = disc(y.detach())
        d_x = disc(x)

        loss_gene = torch.mean((d_y-1)**2)
        loss_disc = torch.mean((d_x-1)**2) + torch.mean(d_z**2)

        result = {
            'loss_gene':loss_gene.item(),
            'loss_disc':loss_disc.item(),
        }
        return [loss_gene, loss_disc], result

    upd = MultilossUpdater(loss_func =_loss_func,
                           optimizers=optimizers,
                           tag       =tag,
                           *arg, **karg)
    
    return upd



def example(generater:nn.Module,
            save_to  :str,
            device   :torch.device):
    n_img     = 8
    val_noise = torch.randn((n_img, 128), dtype=torch.float32, device=device)

    if not os.path.exists(save_to):
        os.mkdir(save_to)

    def _func():
        with torch.no_grad():
            z = val_noise.to(device)
            y = generater(z)

        imgs = y.detach().cpu().numpy()*255
        imgs = imgs.reshape(n_img, 28, 28).astype(np.uint8)

        for i, img in enumerate(imgs):
            path = os.path.join(save_to, str(i)+'.png')
            img  = Image.fromarray(img)
            img.save(path)

    return _func



def main():
    trainset = MNIST(root     ='./mnist',
                     train    =True,
                     download =True,
                     transform=ToTensor())
    valset   = MNIST(root     ='./mnist',
                     train    =False,
                     download =True,
                     transform=ToTensor())

    train_loader = DataLoader(trainset,
                              batch_size =128,
                              shuffle    =True,
                              num_workers=2)
    val_loader   = DataLoader(valset,
                              batch_size =128,
                              shuffle    =False,
                              num_workers=2)

    device = torch.device('cuda:0')

    gene = Generater()
    disc = Discriminater()
    opt_g = optim.Adam(gene.parameters(), lr=5e-5, betas=(0.5, 0.99))
    opt_d = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.99))
    gene.to(device)
    disc.to(device)

    train_updater = GANUpdater(gene, disc, tag='tr', optimizers=[opt_g, opt_d])
    val_updater   = GANUpdater(gene, disc, tag='vl')

    trainer = Trainer(loader =train_loader,
                      updater=train_updater,
                      device =device,
                      save_to='./result')

    trainer.log_report(keys   =[
                                    'tr/loss_gene',
                                    'vl/loss_gene',
                                    'tr/loss_disc',
                                    'vl/loss_disc'
                               ],
                       plots  ={
                           'loss.png':[
                                        'tr/loss_gene',
                                        'vl/loss_gene',
                                        'tr/loss_disc',
                                        'vl/loss_disc'
                                      ],
                           },
                       trigger=(1, 'epoch'))
    
    trainer.add_evaluation(loader =val_loader,
                           updater=val_updater,
                           trigger=(1, 'epoch'))
    
    trainer.save_model(path   ='models/generater_{epoch}.pth',
                       model  =gene,
                       trigger=(1, 'epoch'))
    trainer.save_model(path   ='models/discriminater_{epoch}.pth',
                       model  =disc,
                       trigger=(1, 'epoch'))
                       
    trainer.save_trainer(path   ='trainer.pth',
                         models ={
                                    'generater':gene,  'discriminater':disc,
                                    'opt_g'    :opt_g, 'opt_d'        :opt_d
                         },
                         trigger=(1, 'epoch'))

    trainer.add_event(example(gene, './example', device),
                      trigger=(1, 'epoch'))
    
    trainer.run()


if __name__ == "__main__":
    main()
    