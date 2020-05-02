from typing import Dict, List, Tuple

from torch       import nn, Tensor
from torch.optim import Optimizer



def StanderdUpdater(loss_func:Tuple[Tensor, Dict[str, float]],
                    optimizer:Optimizer=None,
                    tag      :str=None,
                    clip_grad:Tuple[nn.Module, float]=None):
    tag = tag + '/' if tag else ''

    def _func(*arg, **karg) -> Dict[str, float]:
        loss, result = loss_func(*arg, **karg)

        dst = {}
        for k, v in result.items():
            k = tag + k
            if isinstance(v, Tensor):
                v = v.item()
            dst[k] = v
        
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(*clip_grad)
            optimizer.step()

        return dst

    return _func



def MultilossUpdater(loss_func :Tuple[Tensor, Dict[str, float]],
                     optimizers:List[Optimizer]=None,
                     tag       :str=None,
                     clip_grad :List[Tuple[nn.Module, float]]=None):
    tag = tag + '/' if tag else ''
    if optimizers and clip_grad is None:
        clip_grad = [None] * len(optimizers)

    def _func(*arg, **karg) -> Dict[str, float]:
        losses, result = loss_func(*arg, **karg)

        dst = {}
        for k, v in result.items():
            k = tag + k
            if isinstance(v, Tensor):
                v = v.item()
            dst[k] = v
        
        if optimizers:
            for l, o, c in zip(losses, optimizers, clip_grad):
                o.zero_grad()
                l.backward()
                if c:
                    nn.utils.clip_grad_norm_(*c)
                o.step()

        return dst
    
    return _func
