import time
from typing import Callable, Dict, Tuple

from torch import Tensor
from torch.utils.data import DataLoader

from .collection import Collecter
from .console    import ConsoleWriter



class Trainer():
    def __init__(self,
                 loader :DataLoader,
                 updater:Callable[[...], Dict[str, Tensor]],
                 device :torch.device):
        self.loader  = loader
        self.updater = updater
        self.device  = device

        self.iteration  = 0
        self.epoch      = 0
        self.total_time = 0
        self.start_time = time.time()

        self.collecter = Collecter()
        self.keys      = []
        self.writer    = None

        self.print_trigger = 1

        self.epoch_events     = []


    def print_report(self, keys, progress=True, trigger_epoch=1):
        self.keys = keys
        self.collecter = Collecter(keys)


    def add_evaluation(self,
                       loader:DataLoader,
                       updater:Callable[[...], Dict[str, float]],
                       trigger:Tuple[int, str]=(1, 'epoch')):
        def _func():
            length = len(loader)
            for i, vs in enumerate(loader):
                if isinstance(vs, Tensor):
                    vs = [vs.to(self.device)]
                else:
                    vs = [v.to(self.device) for v in vs]
                results = updater(*vs)
                results = self.collecter(results)

                results['epoch']     = self.epoch
                results['iteration'] = self.iteration
                results['time']      = int(time.time() - self.start_time)
                results['progress']  = (i+1) / length

                if self.writer:
                    self.writer(results)
        
        trigger_time, trigger_type = trigger
        {
            'epoch'    :self.epoch_events,
            'iteration':self.iteration_events
        }[trigger_type].append((trigger_time, _func))


    def run_epoch(self):
        self.epoch += 1
        length = len(loader)
        for i, vs in enumerate(self.loader):
            iteration += 1
            if isinstance(vs, Tensor):
                vs = [vs.to(self.device)]
            else:
                vs = [v.to(self.device) for v in vs]
            
            results = self.updater(*vs)
            results = self.collecter(results)

            results['epoch']     = self.epoch
            results['iteration'] = self.iteration
            results['time']      = int(time.time() - self.start_time)
            results['progress']  = (i+1) / length

            if self.writer:
                self.writer(results)

        for trigger, func in self.epoch_events:
            if self.epoch % trigger == 0:
                func()

    def run(self):

