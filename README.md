# stock
NN for classification

## install
```shell
make install        #   Makefile
./script/install.sh #   shell
#   either of two can do
```

## train
```shell
#   1.  change config.py to set the right path of csv data
#       only need to change train_csv, test_csv
#       change input_shape only if the shape of features changed
#   2.  see python3 ./train.py -h for details

python3 train.py
make train  #   use default arguments
```
## requirement
```
python 3.5+
numpy
IPython #   debug
csv
pickle
tensorflow 1.0+
tflearn 0.3
```

## tensorboard
```shell
tensorboard --logdir=./train_log/[YOUR FILE NAME]/tflearn_logs/
make board  #   or Makefile

#   e.g.
tensorboard ./train_log/dnn/tflearn_logs/
#   then open in browser http://localhost:6006/


```

