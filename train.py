import os, argparse
import numpy as np
from IPython import embed

import utils, data_provider, model
from config import cfg

utils.mkdir(cfg.train_log)
utils.mkdir(cfg.cache_path)

def get_args():
    parse = argparse.ArgumentParser(description='dnn for classification')
    parse.add_argument('-c', '--checkpoint', type=str, default=None)
    parse.add_argument('-e', '--epoch', type=int, default=8000)
    parse.add_argument('-s', '--snapshotstep', type=int, default=10)
    parse.add_argument('-b', '--batchsize', type=int, default=-1)
-    parse.add_argument('-v', '--valbatchsize', type=int, default=-1)
    parse.add_argument('-r', '--runid', type=str, default='dnn')
    parse.add_argument('-lr', '--learningrate', type=float, default=1e-3)
    return parse.parse_args()

args = get_args()
dp = data_provider.data_provider()

train_X, train_Y = utils.list_to_array(dp.train[0])
train_W = dp.train[1]

val_X, val_Y = utils.list_to_array(dp.val[0])
val_W = dp.val[1]

if args.batchsize > 0:
    cfg.train_batch_size = args.batchsize
if args.valbatchsize > 0:
    cfg.validation_batch_size = args.valbatchsize

# embed()

net = model.network(checkpoint=args.checkpoint, learning_rate=args.learningrate, name=args.runid)

print('[OPR] start train..')
net.model.fit(
    {'features': train_X, 'weights': train_W}, train_Y, 
    validation_set=[{'features': val_X, 'weights': val_W}, val_Y],
    n_epoch=args.epoch, show_metric=True,
    snapshot_step=args.snapshotstep,
    batch_size=cfg.train_batch_size,
    validation_batch_size=cfg.validation_batch_size,
    run_id=args.runid,
    snapshot_epoch=False,
)
print('[SUC] train done..')

