'''Complete replica of the ICCV 2015 PoseNet Paper'''
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
from keras import regularizers
from keras.initializers import RandomNormal
import gc
import keras.backend as K
import keras_pose_resnet

use_dummy_ds = True
batch_size = 20
num_epochs = 5
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensormain()
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
##################
    tower_1 = Conv2D(64, (1, 1), padding='valid', activation='relu')(x)
    tower_1 = Conv2D(64, (3, 3), padding='valid', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='valid', activation='relu')(x)
    tower_2 = Conv2D(64, (5, 5), padding='valid', activation='relu')(tower_2)

    tower_3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(x)
    tower_3 = Conv2D(64, (1, 1), padding='valid', activation='relu')(tower_3)
    if(K.image_dim_ordering =='tf'):
        bn_axis = 3
    else:
        bn_axis = 1
    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=bn_axis)
    #######################

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

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def inception_net(input_img, t0_f0=64, t1_f0=96, t1_f1=128, t2_f0=16,
t2_f1=32, t3_f1=32):
    tower_0 = Conv2D(t0_f0, (1,1), #padding='same',
    use_bias = True, activation='relu',kernel_regularizer=regularizers.l2(0.01))(input_img)

    tower_1 = Conv2D(t1_f0, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_regularizer=regularizers.l2(0.01))(input_img)
    tower_1 = ZeroPadding2D((1,1))(tower_1)
    tower_1 = Conv2D(t1_f1, (3, 3), #padding='same',
    use_bias = True, activation='relu',kernel_regularizer=regularizers.l2(0.01))(tower_1)

    tower_2 = Conv2D(t2_f0, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_regularizer=regularizers.l2(0.01))(input_img)
    tower_2 = ZeroPadding2D((2,2))(tower_2)
    tower_2 = Conv2D(t2_f1, (5, 5), #padding='same',
    use_bias = True, activation='relu',kernel_regularizer=regularizers.l2(0.01))(tower_2)

    tower_3 = ZeroPatx_3 = Dense(3, activation='tanh', name='tx_3', kernel_regularizer=regularizers.l2(0.01))(y)
    rx_3 = Dense(4, activation='tanh', name='rx_3', kernel_regularizer=regularizers.l2(0.01))(y)
dding2D((1,1))(input_img)
    tower_3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), #padding='same'
    )(tower_3)
    tower_3 = Conv2D(t3_f1, (1, 1), #padding='same',
    use_bias = True, activation='relu',kernel_regularizer=regularizers.l2(0.01))(tower_3)

    if(K.image_dim_ordering =='th'):
        bn_axis = 1
    else:
        bn_axis = 3

    output = layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=bn_axis)
    return (output)

def inc_pose_net():
    img_rows, img_cols, img_channels = 224, 224, 3
    img_input = Input(shape=(img_rows, img_cols, img_channels))
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    # inception_net like architecture with BatchNormalization
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64, (7, 7), strides = 2, activation='relu',
    kernel_regularizer=regularizers.l2(0.01), name='conv1')(x)
    x = MaxPooling2D((3,3), strides=(2,2), name='MaxPooling2D')(x)
    x = BatchNormalization(axis = bn_axis, name='bn_1')(x)

    x = Conv2D(64,(1,1), activation='relu',
    kernel_regularizer=regularizers.l2(0.01))(x) # Xavier
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(192, (3, 3), activation= 'relu',
    kernel_regularizer=regularizers.l2(0.01), name = 'conv2')(x)
    x = BatchNormalization(axis = bn_axis, name='bn_2')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = inception_net(x, 64, 96, 128, 16, 32, 32) # 1
    x = inception_net(x, 128, 128, 192, 96, 64) # 2
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    op1 = inception_net(x, 192, 96, 208, 16, 48, 64) # 3
    # First classification branch
    y = AveragePooling2D(pool_size=(5,5), strides=(3,3))(op1)
    y = Conv2D(128, (1,1), use_bias= True, activation='relu',
    kernel_regularizer=regularizers.l2(0.01))(y)
    y  = Flatten()(y)
    y = Dense(1024, use_bias= True, name='cls1_fc1_pose',
    activation='relu')(y)
    y = Dropout(0.7)(y)
    tx_1 = Dense(3, name='tx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_1 = Dense(4, name='rx_1', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    x = inception_net(op1, 160, 112, 224, 24, 64, 64) # 4
    x = inception_net(x, 128, 128, 256, 24, 24, 64) # 5
    op2 = inception_net(x, 112, 144, 288, 32, 64, 64) # 6
    y  = AveragePooling2D(pool_size=(5,5), strides=(3,3))(op2)
    y = Conv2D(128, (tx_3 = Dense(3, activation='tanh', name='tx_3', kernel_regularizer=regularizers.l2(0.01))(y)
    rx_3 = Dense(4, activation='tanh', name='rx_3', kernel_regularizer=regularizers.l2(0.01))(y)
1,1), use_bias= True, activation='relu',
    kernel_regularizer=regularizers.l2(0.01))(y)
    y  = Flatten()(y)
    y = Dense(1024, use_bias= True, name='cls2_fc1_pose',
    activation='relu')(y)
    y = Dropout(0.7)(y)
    tx_2 = Dense(3, name='tx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_2 = Dense(4, name='rx_2', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    x = inception_net(op2, 256, 160, 320, 32, 128, 128) #7
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = inception_net(x, 256, 160, 320, 32, 128, 128) # 8
    x = inception_net(x, 384, 192, 384, 48, 128, 128) #9

    op3 = AveragePooling2D((5,5), name ='AveragePooling')(x)
    y  = Flatten()(op3)
    y  = Dense(2048, use_bias = True, name='cls3_fc1_pose',
    activation='relu')(y)
    y = Dropout(0.5)(y)

    tx_3 = Dense(3, name='tx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.5), kernel_regularizer=regularizers.l2(0.01))(y)
    rx_3 = Dense(4, name='rx_3', use_bias = True, kernel_initializer=
    RandomNormal(mean=0.0, stddev=0.01), kernel_regularizer=regularizers.l2(0.01))(y)

    model = Model(inputs=img_input, outputs=[tx_1, rx_1, tx_2, rx_2, tx_3, rx_3])

    model.compile(optimizer='rmsprop', loss='mse', loss_weights = [0.25, 100.0, 0.5, 200, 1.0, 400])

    print(model.summary())tx_3 = Dense(3, activation='tanh', name='tx_3', kernel_regularizer=regularizers.l2(0.01))(y)
    rx_3 = Dense(4, activation='tanh', name='rx_3', kernel_regularizer=regularizers.l2(0.01))(y)


    return(model)
def main():
    if(use_dummy_ds == True):
        print('Using dummy ds')
        train_imgs, train_pose_tx, train_pose_rt, test_imgs,test_pose_tx, test_pose_rt=keras_pose_resnet.create_dummy_ds()
    else:
        train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,test_pose_rt = keras_pose_resnet.load_train_test_splits()

    model = inc_pose_net()

    tb = TensorBoard(log_dir='./logs_feature_detector_with_val', histogram_freq=1,
    write_graph=True, write_images=True)
    model.fit(train_imgs, [train_pose_tx, train_pose_rt, train_pose_tx, train_pose_rt,train_pose_tx, train_pose_rt],
    batch_size= batch_size, callbacks = [tb], epochs = num_epochs, shuffle=True)

    model.save('inc_pose_net.h5')

    p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict(test_imgs)
    #p_tx_1, p_rx_1 = model.predict(test_imgs)
    print(p_tx_1)
    K.clear_session()
if __name__ == '__main__':
    main()
