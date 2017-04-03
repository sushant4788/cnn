from __future__ import print_function
#from keras.preprocessing.image import ImageDataGenerator
from imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import TensorBoard
from PIL import Image
from resnet50 import ResNet50
from keras.preprocessing import image
from keras import optimizers
import keras.backend as K
import tensorflow as tf
import numpy as np
import gc
import glob, os
import keras_pose_resnet
from keras.models import load_model
base_dir = base_dir = '/home/sushant/Downloads/Kings/'
img_rows, img_cols, img_channels = 256, 455, 3
model_dir = '/home/sushant/keras_examples/mymodel.h5'
#'/home/sushant/model_and_results/inc_pose_net.h5'
with tf.device('/cpu:0'):
    # model = load_model(model_dir)
    test_imgs, test_pose_tx, test_pose_rt = keras_pose_resnet.test_only_splits(
    base_dir, img_rows, img_cols, img_channels)

    #print(model.summary())
    #mp_tx_1, mp_rx1, mp_tx_2, mp_rx2, mp_tx_3, mp_rx_3  = model.predict(test_imgs)

    mp_tx_3 = np.ones(test_pose_tx.shape)
    mp_rx_3 = np.ones(test_pose_rt.shape)

    tx_er = np.absolute(mp_tx_3 - test_pose_tx)
    rx_er = np.absolute(mp_rx_3 - test_pose_rt)

    print(np.mean(tx_er, axis =0))
    print(np.mean(rx_er, axis =0))
