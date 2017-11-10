train:
	python3 ./train.py 

install:
	./script/install.sh
clean:
	./script/clean.sh
board:
	tensorboard --logdir=./train_log/
