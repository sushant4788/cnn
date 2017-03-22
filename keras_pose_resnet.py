from __future__ import print_function
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Activation, Flatten
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

# Set some hardcoded sizes
img_rows ,img_cols, img_channels = 224, 224, 3
# Some dir locations
training_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_train_mod.txt'
testing_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_test_mod.txt'
dataset_location ='/home/sushant/Downloads/Kings/KingsCollege/'

batch_size = 16
num_epochs = 4
def custom_loss(y_pred, y_truth):
    # ============================================================
    # Define the mean squared error :
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
    # ============================================================
    beta_val = 100
    # ============================================================
    # seperate the first 3 elems
    pos_pred = y_pred[:3]
    #print('pos pred ', K.eval(pos_pred))
    rot_pred = y_pred[3:]
    pos_truth = y_truth[:3]
    rot_truth = y_truth[3:]
    beta = tf.constant(beta_val, dtype='float32',name='beta')
    pos_loss = mean_squared_error(pos_truth, pos_pred)
    #print('pos_loss : \n', K.eval(pos_loss))
    #print(pos_loss)
    rot_loss = mean_squared_error(rot_truth, rot_pred)
    #print(rot_loss)
    #print('rot_loss : \n', K.eval(rot_loss))
    #print(K.eval(beta*rot_loss))
    return(pos_loss + beta*rot_loss)

def process_dataset(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = [x.strip('\n') for x in lines]
    pose = np.zeros((len(lines), 7), dtype='float32')
    print(pose.shape)
    img_list = []
    for i in range(0, len(lines)):
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

def main():
    training_img_list, training_pose = process_dataset(training_file)
    training_pose = training_pose.astype('float32')
    testing_img_list, testing_pose = process_dataset(testing_file)
    testing_pose = testing_pose.astype('float32')
    train_imgs = read_images(training_img_list)
    test_imgs = read_images(testing_img_list)

    train_imgs = train_imgs.astype('float32')
    test_imgs = test_imgs.astype('float32')
    train_imgs /= 255
    test_imgs /= 255
    # import the resnet model
    base_model = ResNet50(weights='imagenet')
    x = base_model.output
    x = Dense(32, activation='relu')(x)
    pred = Dense(7,activation='relu')(x)
    model = Model(input=base_model.input, output=pred)
    print(model.summary())
    sgd = optimizers.SGD(lr = 0.00001, decay=1e-6, momentum=0.9, nesterov = True)
    model.compile(optimizer='SGD', loss=custom_loss, metrics=['accuracy'])
    tb = TensorBoard(log_dir='./logs_feature_detector_with_val', histogram_freq=1,
    write_graph=True, write_images=True)
    model.fit(train_imgs, training_pose, batch_size = batch_size, nb_epoch= num_epochs,validation_data= (test_imgs, testing_pose), callbacks= [tb], shuffle=True)

if __name__ == '__main__':
    main()
'''
LEGACY CUSTOM LOSS
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def custom_loss(y_pred, y_truth):
    # seperate the first 3 elems
    pos_pred = y_pred[:3]
    rot_pred = y_pred[3:]
    pos_truth = y_truth[:3]
    rot_truth = y_truth[3:]
    beta = K.constant(100, dtype='float32',name='beta')
    pos_loss = mean_squared_error(pos_truth, pos_pred)
    rot_loss = mean_squared_error(rot_truth, rot_pred)
    return(pos_loss + beta*rot_loss)'''
