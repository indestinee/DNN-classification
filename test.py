import os, argparse
import numpy as np
from IPython import embed
import pickle

import utils, data_provider, model
from config import cfg

utils.mkdir(cfg.result_path)

def get_args():
    parse = argparse.ArgumentParser(description='dnn for classification')
    parse.add_argument(
        '-c', '--checkpoint', type=str,
        default='./train_log/fc/models/fc-50'
    )
    parse.add_argument('-m', '--model', type=str, default='fc', choices=['fc', 'res'])
    return parse.parse_args()

args = get_args()
dp = data_provider.data_provider()

test_X, index = utils.list_to_array_test(dp.test[0])

net = model.network(checkpoint=args.checkpoint, model=args.model)

print('[OPR] start predict..')
y = net.model.predict({'features': test_X})
print('[SUC] train done..')

with open(cfg.data_bin, 'wb') as f:
    pickle.dump([y, index], f)

