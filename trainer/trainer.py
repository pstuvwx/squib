import os
import time
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .collection import AvgCollecter, EMACollecter
from .console    import ConsoleWriter



def send_device(vs, device):
    if device is None:
        return vs

    if isinstance(vs, Tensor):
        vs = [vs.to(device)]
    else:
        vs = [v.to(device) for v in vs]

    return vs



class Trainer():
    def __init__(self,
                 loader :DataLoader,
                 updater:Callable[[], Dict[str, Tensor]],
                 device :torch.device,
                 save_to:str):
        self.loader  = loader
        self.updater = updater
        self.device  = device

        if not os.path.exists(save_to):
            os.mkdir(save_to)
        self.save_to = save_to

        self.iteration  = 0
        self.epoch      = 0
        self.start_time = 0

        self.tr_collecter = EMACollecter()
        self.vl_collecter = AvgCollecter()
        self.writer       = None

        self.log_trigger = None

        self.epoch_events     = []
        self.iteration_events = []

        self._sv_tr =None


    def end_iteration(self):
        for trigger, func in self.iteration_events:
            if self.iteration % trigger == 0:
                func()


    def end_epoch(self):
        for trigger, func in self.epoch_events:
            if self.epoch % trigger == 0:
                func()


    def calc_progress(self, i, length):
        if self.log_trigger is None:
            return 0

        trigger_time, trigger_type = self.log_trigger
        def calc_iteration():
            p = (self.iteration % trigger_time) / trigger_time
            if p == 0:
                p = 1
            return p
        def calc_epoch():
            p = (length * (self.epoch % trigger_time) + i) / \
                (length * trigger_time)
            return p

        p = \
        {
            'iteration':calc_iteration,
            'epoch'    :calc_epoch
        }[trigger_type]()
        return p


    def run_epoch(self):
        self.epoch += 1
        length = len(self.loader)

        for i, vs in enumerate(self.loader):
            self.iteration += 1

            vs = send_device(vs, self.device)

            results = self.updater(*vs)

            if self.writer is not None:
                results = self.tr_collecter(results)
                results['epoch']     = self.epoch
                results['iteration'] = self.iteration
                results['time']      = int(time.time() - self.start_time)
                results['progress']  = self.calc_progress(i+1, length)
                self.writer(results)

            self.end_iteration()

        self.end_epoch()

    
    def run(self):
        trigger_time, trigger_type = self.log_trigger
        {
            'epoch'    :self.epoch_events,
            'iteration':self.iteration_events
        }[trigger_type].append((trigger_time, self.writer.flush))

        if self._sv_tr:
            self.epoch_events.append(self._sv_tr)

        self.start_time = time.time() - \
                          (self.writer.vals['time'] if self.writer else 0)
        self.writer.init()

        while True:
            self.run_epoch()


    def log_report(self, keys:str, trigger:Tuple[int, str]=(1, 'epoch')):
        self.writer      = ConsoleWriter(keys, self.save_to)
        self.log_trigger = trigger


    def add_evaluation(self,
                       loader:DataLoader,
                       updater:Callable[[], Dict[str, float]],
                       trigger:Tuple[int, str]=(1, 'epoch')):
        def _func():
            length = len(loader)
            for i, vs in enumerate(loader):
                vs      = send_device(vs, self.device)
                results = updater(*vs)
    
                if self.writer is not None:
                    results = self.vl_collecter(results, i==0)
                    results['epoch']     = self.epoch
                    results['iteration'] = self.iteration
                    results['time']      = int(time.time() - self.start_time)
                    results['progress']  = (i+1) / length
                    self.writer(results)

        trigger_time, trigger_type = trigger
        {
            'epoch'    :self.epoch_events,
            'iteration':self.iteration_events
        }[trigger_type].append((trigger_time, _func))


    def save_model(self, path:str, model:torch.nn.Module, trigger=(1, 'epoch')):
        path     = os.path.join(self.save_to, path)
        dirctory = os.path.dirname(path)
        if not os.path.exists(dirctory):
            os.mkdir(dirctory)

        p = path.format(iteration=self.iteration, epoch=self.epoch)
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=self.device))
            print('Load:', p)
        
        def _func():
            p = path.format(iteration=self.iteration, epoch=self.epoch)
            torch.save(model.state_dict(), p)
        
        trigger_time, trigger_type = trigger
        {
            'epoch'    :self.epoch_events,
            'iteration':self.iteration_events
        }[trigger_type].append((trigger_time, _func))
    

    def save_optimizer(self, *arg, **karg):
        self.save_model(*arg, **karg)
    

    def save_trainer(self, path   :str,
                           models :Dict[str, torch.nn.Module],
                           trigger:Tuple[int, str]=(1, 'epoch')):
        assert trigger[1] == 'epoch'

        path     = os.path.join(self.save_to, path)
        dirctory = os.path.dirname(path)
        if not os.path.exists(dirctory):
            os.mkdir(dirctory)

        def _func():
            items = {
                'iteration'   :self.iteration,
                'epoch'       :self.epoch,
                'tr_collecter':self.tr_collecter,
                'vl_collecter':self.vl_collecter,
                'writer'      :self.writer if self.writer else 0
            }
            for k, m in models.items():
                items[k] = m.state_dict()
            torch.save(items, path)
        
        if os.path.exists(path) and os.path.isfile(path):
            items             = torch.load(path, map_location=self.device)
            self.iteration    = items['iteration']
            self.epoch        = items['epoch']
            self.tr_collecter = items['tr_collecter']
            self.vl_collecter = items['vl_collecter']
            self.writer       = None if isinstance(items['writer'], int) else \
                                items['writer']

            for k, m in models.items():
                m.load_state_dict(items[k])
        
        self._sv_tr = (trigger[0], _func)
