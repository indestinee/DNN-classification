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
