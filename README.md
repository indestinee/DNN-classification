# DNN-classification
DNN for features classification. Mainly for this <a href='https://challenger.ai/competition/trendsense/subject'>competition</a>

## requirement
```shell
python 3.5+
numpy
IPython #   debug
csv
pickle
tensorflow 1.0+
tflearn 0.3
```


## install
```shell
make install        #   Makefile
./script/install.sh #   shell
#   either of two works
```

## train
```shell
#   1.  change config.py to set the right path of csv data
#       only need to change train_csv, test_csv
#       change input_shape only if the shape of features changed
#   2.  see python3 ./train.py -h for details

python3 train.py    #   either of two works
make train          #   use default arguments
```

## test
```shell
python3 test.py
```

## demo
```shell
see file demo
```

## tensorboard
```shell
make board              #   Makefile
tensorboard ./train_log #   either of two works
#   then open in browser http://localhost:6006/

```

