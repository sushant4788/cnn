# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings
from keras.layers import merge, Input, Dropout
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
#from imagenet_utils import decode_predictions, preprocess_input
from keras.initializers import RandomNormal
from keras import regularizers, optimizers
import h5py, os, datetime, math, sys

use_dummy_ds = False
use_gpu = False
batch_size = 25
num_epochs = 1

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1,( 1, 1),  name=conv_name_base + '2a',kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                      border_mode='same', name=conv_name_base + '2b', kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, ( 1, 1), strides=strides,
                      name=conv_name_base + '2a', kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), border_mode='same',
                      name=conv_name_base + '2b', kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides =strides,
                             name=conv_name_base + '1', kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.

    # Returns
        A Keras model instance.
    '''
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        if include_top:
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, None, None)
    else:
        if include_top:
            input_shape = (224, 224, 3)
        else:
            input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=2, name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # first classfication branch
    y = AveragePooling2D((7, 7), name='avg_pool_1')(x)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(1024, use_bias= True, name='cls1_fc1_pose',activation='tanh')(y)
    tx_1 = Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_1 = Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)


    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')


    # second classfication branch
    y = AveragePooling2D((7, 7), name='avg_pool_2')(x)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(1024, use_bias= True, name='cls2_fc2_pose',activation='tanh')(y)
    tx_2 = Dense(3, name='tx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_2 = Dense(4, name='rx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    #if include_top:
    #    x = Flatten()(x)
    #    x = Dense(1000, activation='softmax', name='fc1000')(x)

    model = Model(inputs = img_input, outputs = [tx_1, rx_1, tx_2, rx_2])
    model.summary()
    return model

def main():
    model_name = 'resnet50_posenet.h5'
    print('Using original ds')
    # Read the hdf5 data
    landmarks_dir = '/home/sushant/landmarks_seperate/'
    dataset_name = sys.argv[1]
    dataset_fullfile = landmarks_dir+sys.argv[1]+'.h5'
    print('Using ', dataset_fullfile)
    f = h5py.File(dataset_fullfile,'r')
    train_imgs = f['train_imgs'][:]
    train_imgs -= train_imgs.mean()
    train_pose_tx = f['train_pose_tx'][:]
    train_pose_rt = f['train_pose_rt'][:]
    test_imgs = f['test_imgs'][:]
    test_pose_tx = f['test_pose_tx'][:]
    test_pose_rt = f['test_pose_rt'][:]
    test_imgs -= train_imgs.mean()

    # Build the model
    model = ResNet50()

    #optimizers
    sgd = optimizers.SGD(lr = 0.000001, momentum = 0.9, decay = 1e-6)

    # Compile hthe model
    model.compile(optimizer=sgd, loss='mse', loss_weights = [ 0.5, 200, 1.0, 400.0])

    # Fit the data
    model.fit(train_imgs, [train_pose_tx, train_pose_rt, train_pose_tx, train_pose_rt],
    batch_size= batch_size, epochs = num_epochs, shuffle=False)

    model.save(model_name)
    p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict(test_imgs)
    results = np.zeros((test_imgs.shape[0], 2), dtype = 'float32')
    for i in range(0, test_imgs.shape[0]):
        q2 = p_rx_3[i,:] / np.linalg.norm(p_rx_3[i,:])
        q1 = test_pose_rt[i, :] / np.linalg.norm(test_pose_rt[i,:])
        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2*np.arccos(d) * 180/math.pi
        error_x = np.linalg.norm(test_pose_tx[i, :] - p_tx_3[i, :])
        results[i, :] = [error_x, theta]
        print ('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
    median_result = np.median(results,axis=0)
    print('Median error meters: ', median_result[0])
    print('Median error degrees: ', median_result[1])
    np.savetxt('resnet50_posenet.txt', results, delimiter=' ')
    print( 'Success!')
if __name__ == '__main__':
    main()
#if __name__ == '__main__':
    #model = ResNet50(include_top=True, weights='imagenet')

    #img_path = 'elephant.jpg'
    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    #print('Input image shape:', x.shape)

    #preds = model.predict(x)
    #print('Predicted:', decode_predictions(preds))
