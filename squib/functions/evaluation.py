import numpy as np
import torch

def as_numpy(x:torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    x = x.detach()
    if x.device.type >= 'cuda':
        x = x.cpu()
    x = x.numpy()
    return x

def accuracy(y:torch.Tensor, t:torch.Tensor) -> float:
    y = as_numpy(y)
    t = as_numpy(t)

    i = np.argmax(y, axis=1)
    c = i == t
    a = float(np.sum(c)) / np.prod(t.shape)
    return a
