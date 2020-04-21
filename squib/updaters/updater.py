from typing import Dict, Tuple

import torch.nn as nn
from torch       import Tensor
from torch.optim import Optimizer

from squib.functions.evaluation import accuracy



class StanderdUpdater():
    def __init__(self,
                 loss_func:Tuple[Tensor, Dict[str, float]],
                 optimizer:Optimizer=None,
                 tag      :str=None):
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.tag       = tag + '/' if tag else ''

    def __call__(self, *arg, **karg) -> Dict[str, float]:
        loss, result = self.loss_func(*arg, **karg)

        dst = {}
        for k, v in result.items():
            k = self.tag + k
            if isinstance(v, Tensor):
                v = v.item()
            dst[k] = v
        
        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return dst



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



def AutoencodeUpdater(encoder,
                      decoder,
                      optimizer=None,
                      tag=None) -> StanderdUpdater:
    mse = nn.MSELoss()

    def _loss_func(x):
        z = encoder(x)
        y = decoder(z)

        loss = mse(x, y)

        result = {
            'loss':loss.item(),
        }
        return loss, result

    upd = StanderdUpdater(loss_func=_loss_func,
                          optimizer=optimizer,
                          tag      =tag)
    
    return upd
