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
# import the functions from keras_pose_resnet.py
import keras_pose_resnet
# Parmas for training
batch_size = 20
num_epochs = 5

img_rows, img_cols, img_channels = 224, 224, 3
base_dir = '/home/sushant/Downloads/Kings/'

use_dummy_ds = True
def main():
    # read the images
    if(use_dummy_ds == True):
        print('Using dummy ds')
        train_imgs, train_pose_tx, train_pose_rt, test_imgs,test_pose_tx, test_pose_rt=keras_pose_resnet.create_dummy_ds()
    else:
        train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,test_pose_rt = keras_pose_resnet.load_train_test_splits(base_dir, img_rows, img_cols, img_channels )

    base_model = ResNet50()
    # the output of the last layer of the resnet
    y = base_model.get_layer('activation_5').output
    y = Flatten()(y)
    y = Dense(512, activation='tanh')(y)
    int_position = Dense(3, activation='tanh')(y)
    int_rotation = Dense(4, activation='tanh')(y)
    x = base_model.output
    x = Dense(1024, activation='tanh')(x)
    final_position = Dense(3, activation='tanh')(x)
    final_rotation = Dense(4, activation='tanh')(x)
    model = Model(input = base_model.input, output=[int_position, int_rotation,
    final_position, final_rotation])

    #print(model.summary())
    model.compile(optimizer='rmsprop', loss='mse', loss_weights = [0.5, 250.0, 1.0, 350.0])

    model.fit(train_imgs, [train_pose_tx, train_pose_rt, train_pose_tx,
    train_pose_rt], batch_size= batch_size, epochs = num_epochs, shuffle=True)

    model.save('resnet50_final_v2.h5')
    ts_pos_pred, ts_rot_pred, final_ts_position, final_ts_rotation = model.predict(test_imgs)
    print(final_ts_position)
    K.clear_session()
if __name__ == '__main__':
    main()
