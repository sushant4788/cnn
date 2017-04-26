'''Stereo posenet using the inception feaures and GRU unit. Extract inception
net features for both the images and concatenate them to regress for the relative
pose translation and rotation'''
from __future__ import print_function
import numpy as np
import warnings
from keras.models import Sequential
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
import inception_v3
use_dummy_ds = False
use_gpu = False
batch_size = 1
num_epochs = 500
def main():
    '''read the dataset'''
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
    print(train_pose_tx.shape)
    print(train_pose_rt.shape)

    #dataset_dir = '/home/sushant/kitti_dataset/'
    #train_file_name = dataset_dir + 'train.h5'
    #test_file_name = dataset_dir + 'test.h5'
    ##Read the train
    #f = h5py.File(train_file_name)
    #train_imgs = f['train_imgs'][:]
    #train_rt = f['train_R'][:]
    #train_tx = f['train_T'][:]
    #f.close()
    ## Read the test
    #f = h5py.File(test_file_name)
    #test_imgs = f['test_imgs'][:]
    #test_rt = f['test_R'][:]
    #test_tx = f['test_T'][:]
    #f.close()
    ## Subtract the train mean from the train and test set
    #train_imgs -= train_imgs.mean()
    #test_imgs -= train_imgs.mean()
    ## Display the number of hte trai and test samples SANITY CHECK
    #print('Train set has: ', train_imgs.shape[0], 'number of samples ')
    #print('Test set has: ', test_imgs.shape[0], 'number of samples ')
    # Model
    model1= inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    orig_op = model1.output
    y = Flatten()(orig_op)
    y = Dense(1024, use_bias= True, name='cls1_fc1_pose',activation='tanh')(y)
    y = Dropout(0.5)(y)
    tx_1 = Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_1 = Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    model = Model(inputs=model1.inputs, outputs=[tx_1, rx_1])
    sgd = optimizers.SGD(lr = 0.000001, momentum = 0.9, decay = 1e-6)
    model.compile(optimizer=sgd, loss='mse', loss_weights = [1.0, 500.0])
    device ='cpu:0'
    #tb_log_dir_name = '/home_logs/'
    with tf.device(device):
        #model = gru_pose_net(img_rows, img_cols, img_channels)
        # Use TensorBoard to generate graphs of loss
        #tb = TensorBoard(log_dir=tb_log_dir_name, histogram_freq=1,
        #write_graph=True, write_images=True)
        model.fit(train_imgs, [train_pose_tx, train_pose_rt],batch_size= batch_size, epochs = num_epochs, shuffle=False)
        model.save(model_name)
        p_tx_3, p_rx_3 = model.predict(test_imgs)
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

    #model.add(GRU(1024, kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
    #y = model.add(Dropout(0.5))
    #model.add(Dense(3, name='tx_3', use_bias = True, kernel_initializer=
    #RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))
    print(model.summary())

    print('================')
if __name__ == '__main__':
    main()
