from typing import Dict

class AvgCollecter():
    def __init__(self, keys=None):
        self.total = {k:0 for k in keys} if keys else {}
        self.count = {k:0 for k in keys} if keys else {}
        self.val   = None

    def __call__(self, result:Dict[str, float], reset=False):
        for k, v in result.items():
            if k not in self.total or reset:
                self.total[k] = 0
                self.count[k] = 0
            self.total[k] += v
            self.count[k] += 1
        
        self.val = {k:self.total[k] / self.count[k] for k in self.total}

        return self.val


class EMACollecter():
    def __init__(self, alpha=0.98):
        self.val   = {}
        self.alpha = alpha

    def __call__(self, result:Dict[str, float]):
        for k, v in result.items():
            self.val[k] = self.val[k] * self.alpha + v * (1 - self.alpha) \
                          if k in self.val else v
        
        return self.val
