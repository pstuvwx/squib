from typing import Callable, Dict, Union

import torch
import torch.nn as nn
from torch       import Tensor
from torch.nn    import Module
from torch.optim import Optimizer

from functions.evaluation import accuracy



class SupervisedUpdater():
    def __init__(self, model       :Module,
                       optimizer   :Optimizer=None,
                       loss_funcs  :Dict[str, 
                                         Callable[[Tensor, Tensor],
                                                   Tensor]]=None,
                       metrix_funcs:Dict[str,
                                         Callable[[Tensor, Tensor],
                                                   Union[Tensor, float]]]=None,
                       loss_weights:Dict[str, float]=None,
                       tag         :str=None):
        self.model        = model
        self.optimizer    = optimizer
        self.loss_funcs   = loss_funcs   if loss_funcs   else {}
        self.metrix_funcs = metrix_funcs if metrix_funcs else {}
        self.loss_weights = loss_weights if loss_weights else {}
        self.tag          = tag + '/'    if tag          else ''


    def __call__(self, x:torch.Tensor,
                       t:torch.Tensor) -> Dict[str, float]:
        y = self.model(x)

        dst = {}

        loss = 0
        for key, func in self.loss_funcs.items():
            l = func(y, t)
            loss += l * self.loss_weights[key] if key in self.loss_weights \
                    else l
            dst[self.tag+key] = l.item()
        
        for key, func in self.metrix_funcs.items():
            v = func(y, t)
            if isinstance(v, Tensor):
                v = v.item()
            dst[self.tag+key] = v
        
        if self.optimizer:
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()

        return dst



def ClassificationUpdater(model, optimizer=None, tag=None) -> SupervisedUpdater:
    loss_funcs = {
        'loss':lambda x: nn.CrossEntropyLoss()
    }
    metrix_funcs = {
        'accuracy':lambda x: accuracy
    }
    upd = SupervisedUpdater(model       =model,
                            optimizer   =optimizer,
                            loss_funcs  =loss_funcs,
                            metrix_funcs=metrix_funcs,
                            tag         =tag)
    
    return upd
