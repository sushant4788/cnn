
from __future__ import print_function
from keras.models import Model, Sequential
from keras.layers import Conv2D, ZeroPadding2D, Dense, Activation, Flatten, MaxPooling2D
from keras.layers import AveragePooling2D, Input
from keras.callbacks import TensorBoard
from PIL import Image
from keras import optimizers
import keras.backend as K
import tensorflow as tf
import numpy as np

img_rows, img_cols, img_channels = 64, 64, 3
# Training params
batch_size = 1
num_epochs = 20

#num_samples = 20

training_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_train_mod.txt'
testing_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_test_mod.txt'
dataset_location ='/home/sushant/Downloads/Kings/KingsCollege/'

def process_dataset(filename):
    with open(filename) as f:
        lines = f.readlines()
    # ============================
    num_samples = len(lines)
    #=============================
    lines = [x.strip('\n') for x in lines]
    pose = np.zeros((num_samples, 7), dtype='float32')
    #print(pose.shape)
    img_list = []
    #for i in range(0, len(lines)):
    for i in range(0, num_samples):
        line_splits = lines[i].split(' ')
        img_list.append(line_splits[0])
        l_pose = []
        for j in range(1, len(line_splits)):
            l_pose.append(line_splits[j])
        np_pose = np.asarray(l_pose)
        pose[i,:] = np_pose
    #print(img_list[0])
    #print(pose[0,:])
    return(img_list, pose)

def read_images(img_list):

    imgs = np.zeros((len(img_list), img_rows, img_cols, img_channels))
    for i in range(0, len(img_list)):
    #for i in range(0, 1):
        loc = dataset_location + img_list[i]
        print(loc)
        im = Image.open(loc)
        np_im = np.asarray(im, dtype='float32')
        y,x,ch = np_im.shape
        c_y = np.floor(y/2)
        c_x = np.floor(x/2)
        s_y = int(np.floor(c_y - (img_rows/2)))
        s_x = int(np.floor(c_x - (img_cols/2)))
        c_im = np_im[s_y: s_y+img_rows, s_x: s_x+ img_cols, :]
        #c_im = np.swapaxes(c_im, 1,2)
        #c_im = np.swapaxes(c_im, 0,1)
        imgs[i, :, :, :] = c_im
    return(imgs)
'''def custom_loss(y_pred, y_truth):
    # ============================================================
    # Define the mean squared error :
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
    # ============================================================
    # beta_val = 100
    # ============================================================
    # seperate the first 3 elems
    pos_pred = y_pred[:3]
    print('pos pred ', K.eval(pos_pred))
    rot_pred = y_pred[3:]
    pos_truth = y_truth[:3]
    rot_truth = y_truth[3:]
    beta = tf.constant(100, dtype='float32',name='beta')
    pos_loss = mean_squared_error(pos_truth, pos_pred)
    print('pos_loss : \n', K.eval(pos_loss))
    print(pos_loss)
    rot_loss = mean_squared_error(rot_truth, rot_pred)
    print(rot_loss)
    print('rot_loss : \n', K.eval(rot_loss))
    print(K.eval(beta*rot_loss))
    return(pos_loss + beta*rot_loss)'''

def main():
    train_img_list, training_pose = process_dataset(training_file)
    training_pose = training_pose.astype('float32')
    train_imgs = read_images(train_img_list)
    train_imgs = train_imgs.astype('float32')
    train_imgs /=255

    test_img_list, test_pose = process_dataset(testing_file)
    test_pose = test_pose.astype('float32')
    test_imgs = read_images(test_img_list)
    test_img = test_imgs.astype('float32')
    test_imgs /= 255
    tr_position = training_pose[:,:3]
    print(tr_position.shape)
    tr_rotation = training_pose[:,3:]
    print(tr_rotation.shape)
    '''model = Sequential()
    model.add(ZeroPadding2D((2,2), input_shape=(img_rows, img_cols, img_channels)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(2,2))

    model.add(ZeroPadding2D((3,3)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(2,2))

    model.add(ZeroPadding2D((3,3)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))

    #x = model.add(Flatten())
    #x = Dense(32, activation='relu')(x)
    #pred = Dense(7, activation='relu')(x)'''

    main_input = Input(shape=(img_rows, img_cols, img_channels), dtype='float32'
    , name='main_input')
    x = ZeroPadding2D((2,2))(main_input)
    x = Conv2D(64,(3,3))(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2))(x)

    x = ZeroPadding2D((3,3))(x)
    x = Conv2D(128,(5,5))(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2))(x)

    x = ZeroPadding2D((3,3))(x)
    x = Conv2D(64,(5,5))(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    position = Dense(3, activation='tanh')(x)
    rotation = Dense(4, activation='tanh')(x)

    model = Model(inputs=main_input, outputs = [position, rotation])
    print(model.summary)
    model.compile(optimizer='SGD', loss='mse', loss_weights = [1.0, 500.0])
    model.fit(train_imgs, [tr_position, tr_rotation], batch_size = batch_size,
    epochs = num_epochs)
    '''t1 = tf.constant([[1,2,3,4, 0,1,0],[5,6,7,8, 1,0,1]], dtype='float32')
    t2 = tf.constant([[1,9,3,4, 1,1,0],[5,6,7,8, 1,0,1]], dtype='float32')
    print(K.eval(custom_loss(t1,t2)))'''
if __name__ == '__main__':
    main()
