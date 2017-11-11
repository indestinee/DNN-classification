import os, random   
import numpy as np
from IPython import embed

import utils
from config import cfg


            
class data_provider(object):
    def __init__(self, config=cfg, cache=True):
        if not cache or not os.path.isfile(cfg.data_cache):
            self.train, self.val = self.train_val_split(utils.load_csv(cfg.train_csv), 0.9)
            self.test = utils.load_csv(cfg.test_csv, shuffle=False)
            utils.save_cache([self.train, self.val, self.test], cfg.data_cache)
        else:
            self.train, self.val, self.test = utils.load_cache(cfg.data_cache)

    def train_val_split(self, data, rate):
        x, y = data
        nx, ny = int(rate * len(x)), int(rate * len(y))
        return [x[:nx], y[:ny]], [x[nx:], y[ny:]]
    
    def get_train_sample(self, num):
        x = random.sample(self.train[0], num)
        return utils.list_to_array(x)

if __name__ == '__main__':
    dp = data_provider()
    random.seed(23333)
    x = dp.get_train_sample(10)

        

