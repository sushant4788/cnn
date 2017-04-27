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
#import inception_v3
use_dummy_ds = False
use_gpu = False
batch_size = 1
num_epochs = 10
def base_features(img_input):
    if(K.image_dim_ordering =='th'):
        bn_axis = 1
    else:
        bn_axis = 3
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7, 7), strides = 2, activation='relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01),
    use_bias = True)(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    x = BatchNormalization(axis = bn_axis)(x)

    x = Conv2D(64,(1,1), activation='relu',
    kernel_initializer = 'glorot_normal', use_bias = True, kernel_regularizer=regularizers.l2(0.01))(x) # Xavier
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(192, (3, 3), activation= 'relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev = 0.02), use_bias = True, kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization(axis = bn_axis)(x)
    #x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    #model = Model(inputs = img_input, outputs = x)
    #return(model)
    return(x)
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

def stereo_inception_posenet(img_rows=224, img_cols=224, img_channels=1):
    '''Siamese stereo pose net for rot and trans regression using GRU '''
    c_img_input = Input(shape=(img_rows, img_cols, img_channels))
    p_img_input = Input(shape=(img_rows, img_cols, img_channels))
    if(K.image_dim_ordering() == 'th'):
        bn_axis = 1
    else:
        bn_axis = 3
    c_features = base_features(c_img_input)
    p_features = base_features(p_img_input)
    # Combine the feaures
    comb_features = layers.concatenate([c_features, p_features], axis = bn_axis)
    # Build the inception net
    x = inception_net(comb_features, 64, 96, 128, 16, 32, 32) # 1
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
    model = Model(inputs=[c_img_input, p_img_input], outputs=[tx_1, rx_1, tx_2, rx_2, tx_3, rx_3])
    sgd = optimizers.SGD(lr = 0.000001, momentum = 0.9, decay = 1e-6)
    model.compile(optimizer=sgd, loss='mse', loss_weights = [0.25, 125, 0.5, 250, 1.0, 500.0])
    print(model.summary())
    return(model)
def main():
    #'''read the dataset'''
    #print('Using original ds')
    ## Read the hdf5 data
    #landmarks_dir = '/home/sushant/landmarks_seperate/'
    #dataset_name = sys.argv[1]
    #dataset_fullfile = landmarks_dir+sys.argv[1]+'.h5'
    #print('Using ', dataset_fullfile)
    #f = h5py.File(dataset_fullfile,'r')
    #train_imgs = f['train_imgs'][:]
    #train_imgs -= train_imgs.mean()
    #train_pose_tx = f['train_pose_tx'][:]
    #train_pose_rt = f['train_pose_rt'][:]
    #test_imgs = f['test_imgs'][:]
    #test_pose_tx = f['test_pose_tx'][:]
    #test_pose_rt = f['test_pose_rt'][:]
    #test_imgs -= train_imgs.mean()
    #print(train_pose_tx.shape)
    #print(train_pose_rt.shape)

    dataset_dir = '/home/sushant/kitti_dataset/'
    train_file_name = dataset_dir + 'train.h5'
    test_file_name = dataset_dir + 'test.h5'
    #Read the train
    f = h5py.File(train_file_name)
    train_imgs = f['train_imgs'][:]
    train_p_imgs = f['train_p_imgs'][:]
    train_rt = f['train_R'][:]
    train_tx = f['train_T'][:]
    f.close()
    # Read the test
    f = h5py.File(test_file_name)
    test_imgs = f['test_imgs'][:]
    test_p_imgs = f['test_p_imgs'][:]
    test_rt = f['test_R'][:]
    test_tx = f['test_T'][:]
    f.close()
    # Subtract the train mean from the train and test set
    train_imgs -= train_imgs.mean()
    test_imgs -= train_imgs.mean()
    # Display the number of hte trai and test samples SANITY CHECK
    print('Train set has: ', train_imgs.shape[0], 'number of samples ')
    print('Test set has: ', test_imgs.shape[0], 'number of samples ')


    # Choose whether to use CPU or GPU
    if(use_gpu == True):
        device = 'gpu:0'
        print('============USING GPU=========')
    else:
        device = 'cpu:0'
        print('============USING CPU=========')

    # name of the model
    c_time = datetime.datetime.now()
    c_time = c_time.__str__()
    c_time = c_time.replace(" ", "-")
    c_time = c_time.replace(":", '-')
    c_time = c_time.replace(".", '-')
    model_name = 'posenet_'+ c_time + '.h5'
    print('Model name : ', model_name)

    with tf.device(device):
        #model = inc_pose_net(img_rows, img_cols, img_channels)
        # inception model for stereo pose
        img_rows, img_cols, img_channels = 224, 224 ,1
        model = stereo_inception_posenet(img_rows, img_cols, img_channels)

        # Use TensorBoard to generate graphs of loss
        #tb = TensorBoard(log_dir=tb_log_dir_name, histogram_freq=1,
        #write_graph=True, write_images=True)
        # Use the Early stopping
        early_stopping = EarlyStopping(monitor='loss', patience=10, mode = 'auto')
        model.fit([train_imgs, train_p_imgs], [train_tx, train_rt, train_tx, train_rt,train_tx, train_rt],
        batch_size= batch_size, callbacks = [early_stopping], epochs = num_epochs, shuffle=False)
        model.save(model_name)
        p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict([test_imgs, test_p_imgs])
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
def legacy():
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
    with tf.device(device):
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
    print(model.summary())
    print('================')

if __name__ == '__main__':
    main()
