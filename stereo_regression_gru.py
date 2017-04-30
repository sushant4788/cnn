'''Pose regression using GRU'''
from __future__ import print_function
import numpy as np
import warnings
from keras import layers
from keras.layers import merge, Input, Reshape, TimeDistributed, GRU
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers
from keras.initializers import RandomNormal
from Local_Resp_Norm import LRN2D
import h5py, os, datetime, math, sys
import keras.backend as K
import posenet_preprocess
import tensorflow as tf
from PIL import Image

tb_log_dir_name = './stereo_gru_tb_logs'
chpts_dir ='./stereo_gru_checkpoints/'

learning_rate = 0.000001
l_weights = [0.25, 125, 0.5, 250, 1.0, 500.0]
num_epochs = 75
batch_size = 10
use_gpu = True

def get_device_id(use_gpu):
    if(use_gpu == True):
        device = 'gpu:0'
        print('============USING GPU=========')
    else:
        device = 'cpu:0'
        print('============USING CPU=========')
    return(device)

def get_model_name():
    c_time = datetime.datetime.now()
    c_time = c_time.__str__()
    c_time = c_time.replace(" ", "-")
    c_time = c_time.replace(":", '-')
    c_time = c_time.replace(".", '-')
    model_name = 'stereo_reg_gru'+ c_time + '.h5'
    print('Model name : ', model_name)
    return(model_name)

def read_KITTI_ds(dataset_dir):
    '''Load the train adn the test set and reshape the set according to the time
    sequence length'''

    train_file_name = dataset_dir + 'train.h5'
    test_file_name = dataset_dir + 'test.h5'
    #Read the train
    f = h5py.File(train_file_name)
    train_imgs = f['train_imgs'][:]
    train_p_imgs = f['train_p_imgs'][:]
    train_pose_rt = f['train_R'][:]
    train_pose_tx = f['train_T'][:]
    f.close()
    # Read the test
    f = h5py.File(test_file_name)
    test_imgs = f['test_imgs'][:]
    test_p_imgs = f['test_p_imgs'][:]
    test_pose_rt = f['test_R'][:]
    test_pose_tx = f['test_T'][:]
    f.close()
    # Display the number of hte trai and test samples SANITY CHECK
    print('Train set has: ', train_imgs.shape[0], 'number of samples ')
    print('Test set has: ', test_imgs.shape[0], 'number of samples ')
    return(train_imgs, train_p_imgs, train_pose_rt, train_pose_tx,
    test_imgs, test_p_imgs, test_pose_rt, test_pose_tx)
def base_features(img_input):
    if(K.image_dim_ordering =='th'):
        bn_axis = 1
    else:
        bn_axis = 4
    x = TimeDistributed(ZeroPadding2D((3,3)))(img_input)
    x = TimeDistributed(Conv2D(64, (7, 7), strides = 2, activation='relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev=0.015), kernel_regularizer=regularizers.l2(0.01),
    use_bias = True))(x)
    x = TimeDistributed(MaxPooling2D((3,3), strides=(2,2)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    #x = TimeDistributed(BatchNormalization(mode=2))(x)

    x = TimeDistributed(Conv2D(64,(1,1), activation='relu',
    kernel_initializer = 'glorot_normal', use_bias = True, kernel_regularizer=regularizers.l2(0.01)))(x) # Xavier
    x = TimeDistributed(ZeroPadding2D((2,2)))(x)
    x = TimeDistributed(Conv2D(192, (3, 3), activation= 'relu',
    kernel_initializer = RandomNormal(mean=0.0, stddev = 0.02), use_bias = True, kernel_regularizer=regularizers.l2(0.01)))(x)
    x = TimeDistributed(BatchNormalization())(x)
    return(x)

def time_dist_inception(input_img, t0_f0=64, t1_f0=96, t1_f1=128, t2_f0=16,t2_f1=32, t3_f1=32):
    tower_0 = TimeDistributed(Conv2D(t0_f0, (1,1), #padding='same',
    use_bias = True, activation='relu', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))(input_img)

    tower_1 = TimeDistributed(Conv2D(t1_f0, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))(input_img)
    tower_1 = TimeDistributed(ZeroPadding2D((1,1)))(tower_1)
    tower_1 = TimeDistributed(Conv2D(t1_f1, (3, 3), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = RandomNormal(mean=0.0, stddev= 0.04),  kernel_regularizer=regularizers.l2(0.01)))(tower_1)

    tower_2 = TimeDistributed(Conv2D(t2_f0, (1, 1), #padding='same',
    use_bias = True, activation='relu', kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))(input_img)
    tower_2 = TimeDistributed(ZeroPadding2D((2,2)))(tower_2)
    tower_2 = TimeDistributed(Conv2D(t2_f1, (5, 5), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = RandomNormal(mean=0.0, stddev= 0.08), kernel_regularizer=regularizers.l2(0.01)))(tower_2)

    tower_3 = TimeDistributed(ZeroPadding2D((1,1)))(input_img)
    tower_3 = TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), #padding='same'
    ))(tower_3)
    tower_3 = TimeDistributed(Conv2D(t3_f1, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))(tower_3)
    if(K.image_dim_ordering =='th'):
        bn_axis = 1
    else:
        bn_axis = 4
    output = layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=bn_axis)
    return(output)

def stereo_inception_gru(sequence_length = 7, img_rows= 224, img_cols=224, img_channels=1):
    '''Siamese stereo pose net for rot and trans regression using GRU '''
    c_img_input = Input(shape=(sequence_length, img_rows, img_cols, img_channels))
    p_img_input = Input(shape=(sequence_length, img_rows, img_cols, img_channels))
    if(K.image_dim_ordering() == 'th'):
        bn_axis = 1
    else:
        bn_axis = 4
    c_features = base_features(c_img_input)
    p_features = base_features(p_img_input)
    # Combine the feaures
    comb_features = layers.concatenate([c_features, p_features], axis = bn_axis)
    # Build Time Distributed inception_net
    x = time_dist_inception(comb_features, 64, 96, 128, 16, 32, 32)
    x = time_dist_inception(x, 128, 128, 192, 96, 64)
    x = TimeDistributed(MaxPooling2D(pool_size=(3,3), strides = (2,2)))(x)
    op1 = time_dist_inception(x, 192, 96, 208, 16, 48, 64)

    # First GRU branch
    y = TimeDistributed(AveragePooling2D(pool_size=(5,5), strides = (3,3)))(op1)
    y = TimeDistributed(Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))(y)
    y = Reshape((sequence_length, -1))(y)
    y = GRU(1024, return_sequences = True,kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = TimeDistributed(Dropout(0.5))(y)
    tx_1 = TimeDistributed(Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01)))(y)
    rx_1 = TimeDistributed(Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01)))(y)

    x = time_dist_inception(op1, 160, 112, 224, 24, 64, 64)
    x = time_dist_inception(x, 128, 128, 256, 24, 24, 64)
    op2 = time_dist_inception(x, 112, 144, 288, 32, 64, 64)

    # Second GRU branch
    y = TimeDistributed(AveragePooling2D(pool_size=(5,5), strides = (3,3)))(op2)
    y = TimeDistributed(Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01)))(y)
    y = Reshape((sequence_length, -1))(y)
    y = GRU(1024, return_sequences = True,kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = TimeDistributed(Dropout(0.5))(y)
    tx_2 = TimeDistributed(Dense(3, name='tx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01)))(y)
    rx_2 = TimeDistributed(Dense(4, name='rx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01)))(y)

    x = time_dist_inception(op2, 256, 160, 320, 32, 128, 128)
    x = TimeDistributed(MaxPooling2D(pool_size=(3,3), strides =(2,2)))(x)
    x = time_dist_inception(x, 256, 160, 320, 32, 128, 128)
    op3 = time_dist_inception(x, 384, 192, 384, 48, 128, 128)

    # Third GRU branch
    y = TimeDistributed(AveragePooling2D(pool_size=(5,5), strides = (3,3)))(op3)
    y = Reshape((sequence_length, -1))(y)
    y = GRU(2048, return_sequences = True,kernel_initializer = 'glorot_normal', kernel_regularizer=regularizers.l2(0.01))(y)
    y = TimeDistributed(Dropout(0.5))(y)
    tx_3 = TimeDistributed(Dense(3, name='tx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01)))(y)
    rx_3 = TimeDistributed(Dense(4, name='rx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01)))(y)

    # Build the model print summary
    model = Model(inputs=[c_img_input, p_img_input], outputs=[tx_1, rx_1, tx_2, rx_2, tx_3, rx_3])
    model.summary()
    return(model)

def main():

    img_rows, img_cols, img_channels = 224, 224 ,1
    sequence_length = 7
    # load the dataset
    dataset_dir = '/home/sushant/kitti_dataset/'
    train_imgs, train_p_imgs, train_pose_rt, train_pose_tx, test_imgs, test_p_imgs, test_pose_rt, test_pose_tx = read_KITTI_ds(dataset_dir)
    #train_imgs = train_imgs[:1600, :,:,:]
    # =====================Preprocessing the imgs===============================
    # Subtract the train mean from the train and test set
    train_imgs -= train_imgs.mean()
    test_imgs -= train_imgs.mean()
    # ==========================================================================
    # =====================Reshape the dataset =================================
    train_imgs = np.reshape(train_imgs, (-1, sequence_length, img_rows, img_cols, img_channels))
    train_p_imgs = np.reshape(train_p_imgs, (-1, sequence_length, img_rows, img_cols, img_channels))
    train_pose_rt = np.reshape(train_pose_rt, (-1, sequence_length, 4))
    train_pose_tx = np.reshape(train_pose_tx, (-1, sequence_length, 3))
    print(train_imgs.shape)
    # Sanity check
    #img = train_imgs[23,5,:,:,0]
    #print(img.shape)
    #img = np.uint8(img)
    #img = Image.fromarray(img)
    #img.show()

    # build the model
    model = stereo_inception_gru(sequence_length, img_rows, img_cols, img_channels)

    # optimizers
    sgd = optimizers.SGD(lr = learning_rate, momentum = 0.9, decay = 1e-6)

    # Compile the model
    model.compile(optimizer=sgd, loss='mse', loss_weights = l_weights)

    #Callbacks

    # EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', patience=10, mode = 'auto')

    # TensorBoard
    tb = TensorBoard(log_dir=tb_log_dir_name, histogram_freq=1,write_graph=True, write_images=True)

    # Model Check point
    checkpoint = ModelCheckpoint(filepath=chpts_dir + 'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')

    callback_list = [early_stopping, tb, checkpoint]

    # Get the model name
    model_name = get_model_name()

    #Get device
    device = get_device_id(use_gpu)

    # Fit the model
    with tf.device(device):

        # Fit the trainset to teh model
        model.fit([train_imgs, train_p_imgs], [train_pose_tx, train_pose_rt, train_pose_tx, train_pose_rt, train_pose_tx, train_pose_rt],
        batch_size = batch_size, callbacks= callback_list, epochs = num_epochs, shuffle = False, validation_split = 0.1)

        # save the model
        model.save(model_name)

        # Get the prediction
        # p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict([test_imgs, test_p_imgs])

        # Get the pose predictions by adding i, j in loop over the first two indices




if __name__ == '__main__':
    main()
