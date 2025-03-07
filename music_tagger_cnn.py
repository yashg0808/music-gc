# -*- coding: utf-8 -*-
'''MusicTaggerCNN model for Keras.

Reference:

- [Automatic tagging using deep convolutional neural networks](https://arxiv.org/abs/1606.00298)
- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto-tagging-keras)

'''
from __future__ import print_function
from __future__ import absolute_import

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.utils.data_utils import get_file
from keras.layers import Input, Dense


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def MusicTaggerCNN(weights='msd', input_tensor=None):
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        melgram_input = Input(shape=input_tensor)

    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    x = BatchNormalization(axis=time_axis, name='bn_0_freq', trainable=False)(melgram_input)

    x = Convolution2D(32, 3, 3, border_mode='same', name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1', trainable=False)(x)

    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    x = Convolution2D(192, 3, 3, border_mode='same', name='conv4', trainable=False)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4', trainable=False)(x)

    x = Convolution2D(256, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    x = Flatten(name='Flatten_1')(x)

    if weights is None:
        x = Dense(10, activation='sigmoid', name='output')(x)
        model = Model(melgram_input, x)
        return model
    else:
        x = Dense(50, activation='sigmoid', name='output')(x)
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        initial_model = Model(melgram_input, x)
        initial_model.load_weights('weights/music_tagger_cnn_weights_%s.h5' % K._BACKEND,
                                   by_name=True)

        pop_layer(initial_model)

        last = initial_model.get_layer('Flatten_1')
        preds = Dense(10, activation='sigmoid', name='preds')(last.output)
        model = Model(initial_model.input, preds)

        return model
