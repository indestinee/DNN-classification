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
    parse.add_argument('-m', '--model', type=str, default='fc', choices=['fc', 'res'])
    parse.add_argument('-e', '--epoch', type=int, default=300)
    parse.add_argument('-s', '--snapshotstep', type=int, default=10)
    parse.add_argument('-b', '--batchsize', type=int, default=-1)
    parse.add_argument('-v', '--valbatchsize', type=int, default=-1)
    parse.add_argument('-r', '--runid', type=str, default='dnn')
    parse.add_argument('-lr', '--learningrate', type=float, default=1e-3)
    return parse.parse_args()

args = get_args()
dp = data_provider.data_provider()

train_X, train_Y, train_W = *utils.list_to_array_train(dp.train[0]), dp.train[1]
train_W = train_W / np.sum(train_W)

val_X, val_Y, val_W = *utils.list_to_array_train(dp.val[0]), dp.val[1]
val_W = val_W / np.sum(val_W)

if args.batchsize > 0:
    cfg.train_batch_size = args.batchsize
if args.valbatchsize > 0:
    cfg.validation_batch_size = args.valbatchsize

net = model.network(checkpoint=args.checkpoint, learning_rate=args.learningrate, name=args.runid, model=args.model)

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

