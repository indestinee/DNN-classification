import tflearn, os
import tensorflow as tf


import utils
from config import cfg

class network(object):
    def __init__(self, checkpoint=None, learning_rate=1e-3, name='dnn'):
        self.train_log = os.path.join(cfg.train_log, 'dnn')
        utils.mkdir(self.train_log)

        self.models_path = os.path.join(self.train_log, 'models', name)
        utils.mkdir(self.models_path)

        self.tensorboard = os.path.join(self.train_log, 'tflearn_logs')
        self.model = tflearn.DNN(
            self.dnn(learning_rate), checkpoint_path=self.models_path,
            tensorboard_verbose=3, tensorboard_dir=self.tensorboard,
        )

        if checkpoint:
            print('[OPR] loading checkpoint from %s..' % checkpoint)
            self.model.load(checkpoint)
            print('[SUC] checkpoint loaded..')
        
    def fc_layer(self, x, sizes):
        for i, size in enumerate(sizes):
            x = tflearn.fully_connected(
                x, size, activation='relu', regularizer='L2', weight_decay=1e-3, 
                name='fc_layer_%d' % (i+1),
            )
            x = tflearn.layers.normalization.batch_normalization(
                x, name='bn_layer_%d' % (i+1),
            )
        return x

    def dnn(self, learning_rate):
        input_layer = tflearn.input_data(shape=[None, cfg.input_shape])
        
        x = input_layer
        x = self.fc_layer(x, [64, 64, 64, 64])
        x = tflearn.dropout(x, 0.8)

        x = tflearn.fully_connected(x, 2, activation='softmax', name='softmax')

        net = tflearn.regression(
            x, optimizer='adam', loss='categorical_crossentropy',
            learning_rate=learning_rate,
            name='target',
        )

        return net


if __name__ == '__main__':
    net = network()
