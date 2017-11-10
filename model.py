import tflearn, os
import tensorflow as tf
from IPython import embed


import utils
from config import cfg

class network(object):
    def __init__(self, checkpoint=None, learning_rate=1e-3, model='fc', name='dnn'):# {{{
        self.train_log = os.path.join(cfg.train_log, name)
        utils.mkdir(self.train_log)

        self.models_path = os.path.join(self.train_log, 'models')
        utils.mkdir(self.models_path)

        self.checkpoint_path=os.path.join(self.models_path, name)


        self.tensorboard = os.path.join(self.train_log, 'tflearn_logs')
        self.model = tflearn.DNN(
            self.dnn(learning_rate, model=model), 
            checkpoint_path=self.checkpoint_path,
            tensorboard_dir=self.tensorboard,
            tensorboard_verbose=3,
        )

        if checkpoint:
            print('[OPR] loading checkpoint from %s..' % checkpoint)
            self.model.load(checkpoint)
            print('[SUC] checkpoint loaded..')
    # }}}
    def res_block_bn(# {{{
        self, input_layer, channel, depth, 
        name='res_block', regularizer='L1', weight_decay=1e-3, activation='leaky_relu'
    ):
        x = input_layer
        for i in range(depth):
            x = tflearn.fully_connected(
                x, channel, regularizer=regularizer,
                weight_decay=weight_decay,
                name='%s__fc_layer_%d' % (name, i+1),
            )
            x = tflearn.layers.normalization.batch_normalization(
                x, name='%s__bn_layer_%d' % (name, i+1),
            )
            if i + 1 == depth:
                x = x + input_layer
            x = tflearn.layers.core.activation(x, activation=activation, name='%s__activation' % name)
        return x
    # }}}
    def fc_block_bn(# {{{
        self, x, channels, 
        name='fc_block', regularizer='L1', weight_decay=1e-3, activation='leaky_relu'
    ):
        for i, channel in enumerate(channels):
            x = tflearn.fully_connected(
                x, channel, regularizer=regularizer,
                weight_decay=weight_decay,
                name='%s__fc_layer_%d' % (name, i+1),
            )
            x = tflearn.layers.normalization.batch_normalization(
                x, name='%s__bn_layer_%d' % (name, i+1),
            )
            x = tflearn.layers.core.activation(x, activation=activation, name='%s__activation' % name)
        return x
    # }}}
    def fc_block(# {{{
        self, x, channels, 
        name='fc_block', regularizer='L1', weight_decay=1e-3, dropout=0.8, activation='leaky_relu'
    ):
        for i, channel in enumerate(channels):
            x = tflearn.fully_connected(
                x, channel, activation=activation,
                regularizer=regularizer, weight_decay=weight_decay,
                name='%s__fc_layer_%d' % (name, i+1),
            )
            x = tflearn.dropout(x, dropout, name='dropout_layer_%d' % (i+1))
        return x
    # }}}

    def res_demo(self, x):
        x = self.fc_block_bn(x, [32] * 1, name='fc_block')
        for i in range(2):
            x = self.res_block_bn(x, 32, 4, name='res_block_%d' % (i+1))
        return x

    def fc_demo(self, x):
        x = self.fc_block(
            x, [32] * 5,
        )
        return x

    def loss_fun(self, x, y):
        loss = -tf.reduce_sum(self.weights * 
            tf.reduce_sum(y * tf.log(x), axis=1)
        )
        return loss

    def dnn(self, learning_rate, model):# {{{
        input_layer = tflearn.input_data(shape=[None, cfg.input_shape], name='features')
        self.weights = tflearn.input_data(shape=[None], name='weights')
        
        x = input_layer
        if model == 'res':
            x = self.res_demo(x)
        elif model == 'fc':
            x = self.fc_demo(x)

        x = tflearn.dropout(x, 0.5, name='dropout_layer')

        x = tflearn.fully_connected(x, 2, activation='softmax', name='softmax')
        net = tflearn.regression(
            x, optimizer='adam', loss=self.loss_fun,
            learning_rate=learning_rate,
            name='target',
        )

        return net
    # }}}

if __name__ == '__main__':
    net = network()
