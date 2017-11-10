import os, argparse
from IPython import embed

import utils, data_provider, model
from config import cfg

utils.mkdir(cfg.train_log)
utils.mkdir(cfg.cache_path)

def get_args():
    parse = argparse.ArgumentParser(description='dnn for classification')
    parse.add_argument('-c', '--checkpoint', type=str, default=None)
    parse.add_argument('-e', '--epoch', type=int, default=8000)
    parse.add_argument('-b', '--batchsize', type=int, default=20000)
    parse.add_argument('-v', '--valbatchsize', type=int, default=4000)
    parse.add_argument('-s', '--snapshotstep', type=int, default=100)
    parse.add_argument('-r', '--runid', type=str, default='dnn')
    parse.add_argument('-lr', '--learningrate', type=float, default=1e-3)
    return parse.parse_args()

args = get_args()
dp = data_provider.data_provider()

train_set = utils.list_to_array(dp.train[0])
validation_set = utils.list_to_array(dp.val[0])

# embed()

net = model.network(checkpoint=args.checkpoint, learning_rate=args.learningrate, name=args.runid)
net.model.fit(
    *train_set, validation_set=validation_set, n_epoch=args.epoch, show_metric=True,
    batch_size=args.batchsize, snapshot_step=args.snapshotstep,
    validation_batch_size=args.valbatchsize, run_id=args.runid,
    snapshot_epoch=False,
)

