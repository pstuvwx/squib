from typing import Dict, List


class ConsoleWriter():
    def __init__(self,
                 keys:Dict[str, List[str]]):
        self.keys = ['epoch', 'iteration', 'time'] + keys + ['progress']
        self.fmts = ['{:10d}', '{:10d}', '{:10d}'] + \
                    ['{:^ .' + str(min(len(k), 10)-7) + 'e}' for k in keys] + \
                    ['{:^10.3%}']


    def init(self, hist=None):
        for k in self.keys:
            width = min(len(k), 10)
            fmt   = '{:^' + str(width) + 's}'
            print(fmt.format(k), end='  ')
        
        if hist:
            for h in hist:
                self.start()
                self.print_iteration(h)


    def start(self):
        print()


    def print_iteration(self, values):
        print('\r', end='')
        for f, k in zip(self.fmts, self.keys):
            print(f.format(values[k]), end='  ')
        print('', end='', flush=True)
