import csv
import os
from typing import Dict, List, Union

import matplotlib.pyplot as plt


def load_csv(path, encoding='utf_8_sig'):
    with open(path, 'r', newline='', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=',')
        dst = list(reader)
    return dst



def save_csv(path, obj, encoding='utf_8_sig'):
    with open(path, 'w', newline='', encoding=encoding) as f:
        wtr = csv.writer(f)
        wtr.writerows(obj)



def save_plot(path, keys, log, xkey):
    tags  = log[0]
    log   = log[1:]
    xi    = tags.index(xkey)
    xtick = [l[xi] for l in log]
    for k in keys:
        yi   = tags.index(k)
        yval = [l[yi] for l in log]
        plt.plot(xtick, yval, label=k)
    plt.legend()
    plt.grid()

    plt.savefig(path)
    plt.close()



class ConsoleWriter():
    def __init__(self,
                 keys   :List[str],
                 save_to:str=None,
                 plots  :Dict[str, List[str]]=None,
                 xkey   :str='epoch'):
        self.keys = ['epoch', 'iteration', 'time'] + keys + ['progress']
        self.fmts = ['{:^10d}', '{:^10d}', '{:^10d}'] + \
                    ['{: ^ '+str(max(len(k), 10))+'.3e}' for k in keys] + \
                    ['{:^10.3%}']
        self.vals = {k:0 for k in self.keys}

        self.hist = [['epoch', 'iteration', 'time'] + keys]
        self.save = os.path.join(save_to, 'log.csv') \
                    if save_to is not None else None

        self.plot = [(os.path.join(save_to, k), plots[k]) for k in plots] \
                    if plots is not None else []
        self.xkey = xkey


    def init(self):
        for k in self.keys:
            width = max(len(k), 10)
            fmt   = '{:^' + str(width) + 's}'
            print(fmt.format(k), end='  ')

        print()

        for hs in self.hist[1:]:
            hs = dict(zip(self.keys[:-1], hs))
            hs['progress'] = 1
            self.__call__(hs)
            print()
        if len(self.hist) > 1:
            self.vals = dict(zip(self.keys[:-1], self.hist[-1]))


    def __call__(self, values:Dict[str, Union[int, float]]):
        self.vals.update(**values)

        print('\r', end='')
        for f, k in zip(self.fmts, self.keys):
            print(f.format(self.vals[k]), end='  ')
        print('', end='', flush=True)


    def flush(self):
        self.hist.append([self.vals[k] for k in self.keys[:-1]])
        print()
        if self.save:
            save_csv(self.save, self.hist)
            for path, keys in self.plot:
                save_plot(path, keys, self.hist, self.xkey)
