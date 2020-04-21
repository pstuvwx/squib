import csv
import os
from typing import Dict, List, Union


def load_csv(path, encoding='utf_8_sig'):
    with open(path, 'r', newline='', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=',')
        dst = list(reader)
    return dst



def save_csv(path, obj, encoding='utf_8_sig'):
    with open(path, 'w', newline='', encoding=encoding) as f:
        wtr = csv.writer(f)
        wtr.writerows(obj)



class ConsoleWriter():
    def __init__(self,
                 keys:List[str],
                 save_to:str=None):
        self.keys = ['epoch', 'iteration', 'time'] + keys + ['progress']
        self.fmts = ['{:^10d}', '{:^10d}', '{:^10d}'] + \
                    ['{: ^ '+str(max(len(k), 10))+'.3e}' for k in keys] + \
                    ['{:^10.3%}']
        self.vals = {k:0 for k in self.keys}

        self.hist = [['epoch', 'iteration', 'time'] + keys]
        self.save = os.path.join(save_to, 'log.csv')


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
