from typing import Dict

class Collecter():
    def __init__(self, keys=None):
        self.total = {k:0 for k in keys} if keys else {}
        self.count = {k:0 for k in keys} if keys else {}
        self.avg   = None

    def __call__(self, result:Dict[str, float]):
        for k, v in result.items():
            if k not in self.total:
                self.total[k] = 0
                self.count[k] = 0
            self.total[k] += v
            self.count[k] += 1
        
        dst = {k:self.total[k] / self.count[k] for k in self.total}

        self.avg = dst

        return dst
