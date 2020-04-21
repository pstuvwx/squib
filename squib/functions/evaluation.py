import numpy as np
import torch

def as_numpy(x:torch.Tensor):
    if isinstance(x, np.ndarray):
        return x
    x = x.detach()
    if x.device.type >= 'cuda':
        x = x.cpu()
    x = x.numpy()
    return x

def accuracy(y:torch.Tensor, t:torch.Tensor):
    y = as_numpy(y)
    t = as_numpy(t)

    assert len(y.shape) == 2 and len(t.shape) == 1

    i = np.argmax(y, axis=1)
    c = i == t
    a = float(np.sum(c)) / t.shape[0]
    return a
