'''Load the posenet model and test the performance'''
from __future__ import print_function
import numpy as np
import warnings
from keras import layers
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from imagenet_utils import decode_predictions, preprocess_input
from keras.callbacks import TensorBoard
from keras import regularizers, optimizers
from keras.initializers import RandomNormal
from Local_Resp_Norm import LRN2D
from keras.models import load_model
import gc, h5py
import keras.backend as K
import posenet_preprocess
import tensorflow as tf
import datetime


def main():
    '''Load the model '''
    test_file = '/home/sushant/dataset/test.h5'
    model_file = '/home/sushant/keras_pose_resnet/source/models/posenet_2017-04-05-14-15-37-646073.h5'

    # load the model
    model = load_model(model_file, custom_objects={'LRN2D':LRN2D})
    test = h5py.File(ds_dir+test_prefix, 'r')
    test_imgs = test['test_imgs'][:]
    test_pose_tx = test['test_pose_tx'][:]
    test_pose_rt = test['test_pose_rt'][:]
    test.close()

    p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict(test_imgs)
    print(p_tx_3.shape)
if __name__ == '__main__':
    main()
