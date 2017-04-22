'''Complete posenet from ICCV 2015 paper.
This is an attempt to reproduce the results of the the ICCV 2015 paper titled
POSENET :
Following are the changes that are made in the implementation that differ from
the original
1) BatchNormalization is used instead of LocalResponseNorm (mainly because
LRN lacks implementation in Keras as of now : FIXME)
2) Use tanh as activation in the FC layers. This is because the loss was
observed to be Nan, moving to tanh seems to fix this.
'''

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
import h5py, os, datetime, math, sys
import keras.backend as K
import posenet_preprocess
import tensorflow as tf

use_dummy_ds = False
use_gpu = True
batch_size = 75
num_epochs = 500

def inception_net(input_img, t0_f0=64, t1_f0=96, t1_f1=128, t2_f0=16,
t2_f1=32, t3_f1=32):
    '''This is the base building block of the inception net'''
    tower_0 = Conv2D(t0_f0, (1,1), #padding='same',
    use_bias = True, activation='relu', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(input_img)

    tower_1 = Conv2D(t1_f0, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(input_img)
    tower_1 = ZeroPadding2D((1,1))(tower_1)
    tower_1 = Conv2D(t1_f1, (3, 3), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = RandomNormal(mean=0.0, stddev= 0.04), kernel_regularizer=regularizers.l2(0.01))(tower_1)

    tower_2 = Conv2D(t2_f0, (1, 1), #padding='same',
    use_bias = True, activation='relu', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(input_img)
    tower_2 = ZeroPadding2D((2,2))(tower_2)
    tower_2 = Conv2D(t2_f1, (5, 5), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = RandomNormal(mean=0.0, stddev= 0.08), kernel_regularizer=regularizers.l2(0.01))(tower_2)

    tower_3 = ZeroPadding2D((1,1))(input_img)
    tower_3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), #padding='same'
    )(tower_3)
    tower_3 = Conv2D(t3_f1, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(tower_3)

    if(K.image_dim_ordering =='th'):
        bn_axis = 1
    else:
        bn_axis = 3

    output = layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=bn_axis)
    return (output)

def inc_pose_net(img_rows, img_cols, img_channels):
    '''We build the pose regression network by building a model like the GoogleLenet
    and following implementation instructions from the PoseNet paper'''

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
    #x = BatchNormalization(axis = bn_axis, name='bn_1')(x)
    x = LRN2D()(x)

    x = Conv2D(64,(1,1), activation='relu',
    kernel_initializer = 'glorot_normal', use_bias = True, kernel_regularizer=regularizers.l2(0.01))(x) # Xavier
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(192, (3, 3), activation= 'relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev = 0.02), use_bias = True, kernel_regularizer=regularizers.l2(0.01), name = 'conv2')(x)
    #x = BatchNormalization(axis = bn_axis, name='bn_2')(x)
    x = LRN2D()(x)
    #x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = inception_net(x, 64, 96, 128, 16, 32, 32) # 1
    x = inception_net(x, 128, 128, 192, 96, 64) # 2
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    op1 = inception_net(x, 192, 96, 208, 16, 48, 64) # 3
    # First classification branch
    y = AveragePooling2D(pool_size=(5,5), strides=(3,3))(op1)
    y = Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y  = Flatten()(y)
    y = Dense(1024, use_bias= True, name='cls1_fc1_pose',
    # activation='relu'
    activation= 'tanh')(y)
    y = Dropout(0.7)(y)
    tx_1 = Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_1 = Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    x = inception_net(op1, 160, 112, 224, 24, 64, 64) # 4
    x = inception_net(x, 128, 128, 256, 24, 24, 64) # 5
    op2 = inception_net(x, 112, 144, 288, 32, 64, 64) # 6
    # Second classification net
    y  = AveragePooling2D(pool_size=(5,5), strides=(3,3))(op2)
    y = Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y  = Flatten()(y)
    # Mod: Changing the activation to tanh
    y = Dense(1024, use_bias= True, name='cls2_fc1_pose',
    #activation='relu'
    activation='tanh')(y)
    y = Dropout(0.7)(y)
    tx_2 = Dense(3, name='tx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_2 = Dense(4, name='rx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    x = inception_net(op2, 256, 160, 320, 32, 128, 128) #7
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = inception_net(x, 256, 160, 320, 32, 128, 128) # 8
    x = inception_net(x, 384, 192, 384, 48, 128, 128) #9
    # Final classification net
    op3 = AveragePooling2D((5,5), name ='AveragePooling')(x)
    y  = Flatten()(op3)
    # Changing the activation to tanh
    y  = Dense(2048, use_bias = True, name='cls3_fc1_pose',
    # activation='relu'
    activation = 'tanh')(y)
    y = Dropout(0.5)(y)

    tx_3 = Dense(3, name='tx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_3 = Dense(4, name='rx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    # Build the model, print summary !!
    model = Model(inputs=img_input, outputs=[tx_1, rx_1, tx_2, rx_2, tx_3, rx_3])
    sgd = optimizers.SGD(lr = 0.000001, momentum = 0.9, decay = 1e-6)
    model.compile(optimizer=sgd, loss='mse', loss_weights = [0.25, 125, 0.5, 250, 1.0, 500.0])
    print(model.summary())
    return(model)
def main():
    # ==========================================================================
    # initialization Params go here !!
    # ==========================================================================
    num_samples = 20
    #img_rows, img_cols, img_channels = 256, 455, 3
    img_rows, img_cols, img_channels = 224, 224, 3
    base_dir = '/home/sushant/Downloads/Kings/'
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
        location_list = ['OldHospital', 'StMarysChurch', 'KingsCollege', 'Street', 'ShopFacade']
        c_loc = location_list[2]
        print('Training and Testing on the ', c_loc, ' dataset')
        #train = h5py.File(os.path.join(ds_dir, 'train_' + c_loc + '.h5'), 'r')
        #train = h5py.File(os.path.join(ds_dir, 'train.h5'), 'r')
        #train_imgs = train['train_imgs'][:]
        # save the subtract the mean of the training set from the train and the test set
        ts_mean = train_imgs.mean()
        train_imgs = train_imgs-ts_mean
        train_pose_tx = train['train_pose_tx'][:]
        train_pose_rt = train['train_pose_rt'][:]
        train.close()

        test = h5py.File(os.path.join(ds_dir, 'test_' + c_loc + '.h5'), 'r')
        #test = h5py.File(os.path.join(ds_dir, 'test.h5'), 'r')
        test_imgs = test['test_imgs'][:]
        # Subtract the mean
        test_imgs = test_imgs- ts_mean
        test_pose_tx = test['test_pose_tx'][:]
        test_pose_rt = test['test_pose_rt'][:]
        test.close()


    with tf.device(device):
        model = inc_pose_net(img_rows, img_cols, img_channels)
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


'''train = h5py.File(ds_dir+train_prefix, 'r')
train_imgs = train['train_imgs'][:]
train_pose_tx = train['train_pose_tx'][:]
train_pose_rt = train['train_pose_rt'][:]
train.close()

test = h5py.File(ds_dir+test_prefix, 'r')
test_imgs = test['test_imgs'][:]
test_pose_tx = test['test_pose_tx'][:]
test_pose_rt = test['test_pose_rt'][:]
test.close()
# Modify the train and test data
train_imgs = train_imgs[:894,:,:,:]
train_pose_tx = train_pose_tx[:894,:]
train_pose_rt = train_pose_rt[:894,:]
test_imgs = test_imgs[:181, :,:,:]
test_pose_tx = test_pose_tx[:181,:]
test_pose_rt = test_pose_rt[:181,:]

#print(train_imgs[3,:,:,:])'''
