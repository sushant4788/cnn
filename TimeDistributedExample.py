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
from keras.layers import SimpleRNN
from keras.layers import Reshape
import h5py, os, datetime, math, sys
import keras.backend as K
import posenet_preprocess
import tensorflow as tf
import cap6412_resnet_inception
from keras.layers import TimeDistributed
'''Simple GRU on Mnist'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop

#batch_size = 32
#num_classes = 10
#epochs = 200
#hidden_units = 100

#learning_rate = 1e-6
#clip_norm = 1.0

## the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#x_train = x_train.reshape(x_train.shape[0], -1, 1)
#x_test = x_test.reshape(x_test.shape[0], -1, 1)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')
#print('x train shape', x_train.shape)
## convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#print(x_train.shape[1:])
#print('Evaluate IRNN...')
#model = Sequential()
#model.add(SimpleRNN(hidden_units,
#                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
#                    recurrent_initializer=initializers.Identity(gain=1.0),
#                    activation='relu',
#                    input_shape=x_train.shape[1:]))
#model.add(Dense(num_classes))
#model.add(Activation('softmax'))
#rmsprop = RMSprop(lr=learning_rate)
#model.compile(loss='categorical_crossentropy',
#              optimizer=rmsprop,
#              metrics=['accuracy'])

#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test))

#scores = model.evaluate(x_test, y_test, verbose=0)
#print('IRNN test score:', scores[0])
#print('IRNN test accuracy:', scores[1])
use_dummy_ds = False
use_gpu = False
batch_size = 1
num_epochs = 5

def simple_gru(img_rows, img_cols, img_channels):
    #Simple example of the GRU based pose regression

    # Create a simple model
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
    #x = TimeDistributed(cap6412_resnet_inception.inception_net(x, 64, 96, 128, 16, 32, 32)) # 1
    #x = TimeDistributed(MaxPooling2D(pool_size=(3,3), strides=(2,2)))(x)
    #y = Flatten()(x)
    x = Reshape((1,-1))(x)
    y = GRU(128, kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01), input_shape=(x.shape[1:]))(x)
    #y = Dense(1024)(y)
    y = Dropout(0.7)(y)
    tx_1 = Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_1 = Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    model = Model(inputs=img_input, outputs = [tx_1, rx_1])
    sgd = optimizers.SGD(lr = 0.00001, momentum = 0.9, decay = 1e-6)
    model.compile(optimizer=sgd, loss='mse', loss_weights=[1.0, 500.0])
    #print(model.summary())
    return(model)
def another_another(img_rows=224, img_cols=224, img_channels =3):

    img_input = Input(shape=(img_rows, img_cols, img_channels))
    x = Conv2D(32, (7,7), strides =1, activation='relu')(img_input)
    x = Reshape((1,-1))(x)
    hidden_units = 100
    x = GRU(hidden_units, kernel_initializer = initializers.RandomNormal(stddev=0.001), activation='relu')(x)
    model = Model(inputs=img_input, outputs=x)
    model.compile(optimizer='rmsprop', loss='mse')
    print(model.summary())

def another_model(img_rows=224, img_cols=224, img_channels=3):
    model=Sequential()
    model.add(Conv2D(32, (7,7), activation ='relu', input_shape=(img_rows, img_cols, img_channels)))
    #model.add(Flatten())
    model.add(Reshape((1,-1)))
    print(model.summary())
    hidden_units = 100
    model.add(GRU(hidden_units,
                        kernel_initializer=initializers.RandomNormal(stddev=0.001),
                        activation='relu'))
    print(model.summary())
#def main():
    #another_model()
    #another_another()

def main():
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
        print('Train images has :', train_imgs.shape)
    with tf.device(device):
        model = simple_gru(img_rows, img_cols, img_channels)
        # Use TensorBoard to generate graphs of loss
        tb = TensorBoard(log_dir=tb_log_dir_name, histogram_freq=1,
        write_graph=True, write_images=True)
        model.fit(train_imgs, [train_pose_tx, train_pose_rt],
        batch_size= batch_size, callbacks = [tb], epochs = num_epochs, shuffle=False)
        #model.save(model_name)
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

if __name__ == '__main__':
    main()
