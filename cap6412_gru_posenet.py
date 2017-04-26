'''Complete implementation of the pose net using the Googleinception net for
feature extraction and the GRU recurrent unit for pose regression. The input is
the landmarks sequence from the ICCV 2015paper posenet '''

from __future__ import print_function

import numpy as np
import warnings
from keras import layers
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
#from imagenet_utils import decode_predictions, preprocess_input
from keras.callbacks import TensorBoard
from keras import regularizers, optimizers
from keras.initializers import RandomNormal
from Local_Resp_Norm import LRN2D
from keras.layers import GRU
import h5py, os, datetime, math, sys
import keras.backend as K
import posenet_preprocess
import tensorflow as tf
from keras.layers import Reshape

import cap6412_resnet_inception
use_dummy_ds = False
use_gpu = True
batch_size = 75
num_epochs = 500
def gru_pose_net(img_rows, img_cols, img_channels):
    '''Inception net model to be tested on pose regression on monocular images
    using the landmarks images and the GRU unit for pose regression'''
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    # inception_net like architecture with BatchNormalization
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7, 7), strides = 2, activation='relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01),
    use_bias = True, name='conv1')(x)
    x = MaxPooling2D((3,3), strides=(2,2), name='MaxPooling2D')(x)
    x = BatchNormalization(axis = bn_axis, name='bn_1')(x)
    #x = LRN2D()(x)

    x = Conv2D(64,(1,1), activation='relu',
    kernel_initializer = 'glorot_normal', use_bias = True, kernel_regularizer=regularizers.l2(0.01))(x) # Xavier
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(192, (3, 3), activation= 'relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev = 0.02), use_bias = True, kernel_regularizer=regularizers.l2(0.01), name = 'conv2')(x)
    x = BatchNormalization(axis = bn_axis, name='bn_2')(x)
    #x = LRN2D()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = cap6412_resnet_inception.inception_net(x, 64, 96, 128, 16, 32, 32) # 1
    x = cap6412_resnet_inception.inception_net(x, 128, 128, 192, 96, 64) # 2
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    op1 = cap6412_resnet_inception.inception_net(x, 192, 96, 208, 16, 48, 64) # 3
    # First classification branch
    y = AveragePooling2D(pool_size=(5,5), strides=(3,3))(op1)
    y = Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y1 = Reshape((1,-1))(y)
    y1 = GRU(1024,kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y1)
    y2= Flatten()(y)
    y2 = Dense(1024, use_bias= True, name='cls1_fc1_pose',
    activation='tanh')(y2)
    # Remove the Flatten net and add the reshape
    # activation='relu'
    # activation= 'tanh')(y)
    y1 = Dropout(0.7)(y1)
    y2 = Dropout(0.7)(y2)
    tx_1 = Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y1)
    rx_1 = Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y2)

    x = cap6412_resnet_inception.inception_net(op1, 160, 112, 224, 24, 64, 64) # 4
    x = cap6412_resnet_inception.inception_net(x, 128, 128, 256, 24, 24, 64) # 5
    op2 = cap6412_resnet_inception.inception_net(x, 112, 144, 288, 32, 64, 64) # 6
    # Second classification net
    y  = AveragePooling2D(pool_size=(5,5), strides=(3,3))(op2)
    y = Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y1 = Reshape((1,-1))(y)
    y1 = GRU(1024,kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y1)
    y2 = Flatten()(y)
    #y2 = Dense(1024, use_bias= True, name='cls2_fc1_pose',
    #activation='tanh')(y2)

    y2 = Dense(1024, use_bias= True, name='cls2_fc1_pose',
    activation='tanh')(y2)
    y1 = Dropout(0.7)(y1)
    y2 = Dropout(0.7)(y2)
    tx_2 = Dense(3, name='tx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y1)
    rx_2 = Dense(4, name='rx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y2)

    x = cap6412_resnet_inception.inception_net(op2, 256, 160, 320, 32, 128, 128) #7
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = cap6412_resnet_inception.inception_net(x, 256, 160, 320, 32, 128, 128) # 8
    x = cap6412_resnet_inception.inception_net(x, 384, 192, 384, 48, 128, 128) #9
    # Final classification net
    op3 = AveragePooling2D((5,5), name ='AveragePooling')(x)
    y1 = Reshape((1,-1))(op3)
    y2  = Flatten()(op3)
    y1 = GRU(2048,kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y1)
    y2  = Dense(2048, use_bias = True, name='cls3_fc1_pose',activation = 'tanh')(y2)
    y1 = Dropout(0.5)(y1)
    y2 = Dropout(0.5)(y2)
    tx_3 = Dense(3, name='tx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y1)
    rx_3 = Dense(4, name='rx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y1)

    # Build the model, print summary !!
    model = Model(inputs=img_input, outputs=[tx_1, rx_1, tx_2, rx_2, tx_3, rx_3])
    sgd = optimizers.SGD(lr = 0.000001, momentum = 0.9, decay = 1e-6)
    model.compile(optimizer=sgd, loss='mse', loss_weights = [0.25, 125, 0.5, 250, 1.0, 500.0])
    print(model.summary())
    return(model)

def main():
    # Use the landmarks dataset
        # ==========================================================================
        # initialization Params go here !!
        # ==========================================================================
        num_samples = 20
        #img_rows, img_cols, img_channels = 256, 455, 3
        img_rows, img_cols, img_channels = 224, 224, 3
        base_dir = '/home/sushant/Downloads/'
        ds_dir = '/home/sushant/dataset_mod/'
        #ds_dir = '/home/sushant/sep_ds/'
        train_prefix = 'train.h5'
        test_prefix = 'test.h5'
        # GPU or CPU
        if(use_gpu == True):
            device = 'gpu:0'
            print('============USING GPU=========')
        else:
            device = 'cpu:0'
            print('============USING CPU=========')
        # TensorBoard log dir name:
        tb_log_dir_name= './logs_posenet'
        # name of the model
        c_time = datetime.datetime.now()
        c_time = c_time.__str__()
        c_time = c_time.replace(" ", "-")
        c_time = c_time.replace(":", '-')
        c_time = c_time.replace(".", '-')
        model_name = 'posenet_'+ c_time + '.h5'
        print('Model name : ', model_name)
        # ==========================================================================
        # The main part !!
        # ==========================================================================
        if(use_dummy_ds == True):
            print('Using dummy ds')
            model_name = 'dummy_' + model_name
            tb_log_dir_name='./dummy_logs'
            train_imgs, train_pose_tx, train_pose_rt, test_imgs,test_pose_tx,test_pose_rt=posenet_preprocess.create_dummy_ds(num_samples, img_rows,
            img_cols, img_channels)
        else:
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

        with tf.device(device):
            model = gru_pose_net(img_rows, img_cols, img_channels)
            # Use TensorBoard to generate graphs of loss
            tb = TensorBoard(log_dir=tb_log_dir_name, histogram_freq=1,
            write_graph=True, write_images=True)
            model.fit(train_imgs, [train_pose_tx, train_pose_rt, train_pose_tx, train_pose_rt,train_pose_tx, train_pose_rt],
            batch_size= batch_size, callbacks = [tb], epochs = num_epochs, shuffle=False)
            model.save(model_name)
            p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict(test_imgs)
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
            np.savetxt('results.txt', results, delimiter=' ')
            print( 'Success!')
            K.clear_session()
if __name__ == '__main__':
    main()
