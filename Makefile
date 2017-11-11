train:
	python3 ./train.py 
test:
	python3 ./test.py
	python3 ./save.py
install:
	./script/install.sh
clean:
	./script/clean.sh
clear:
	./script/clear.sh
board:
	tensorboard --logdir=./train_log/
