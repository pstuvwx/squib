from typing import Dict, Tuple

from torch       import Tensor
from torch.optim import Optimizer



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
