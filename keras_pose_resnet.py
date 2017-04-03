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

# Set some hardcoded sizes
# img_rows ,img_cols, img_channels = 224, 224, 3
# Some dir locations
#training_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_train_mod.txt'
#testing_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_test_mod.txt'
#dataset_location ='/home/sushant/Downloads/Kings/KingsCollege/'
#base_dir ='/home/sushant/Downloads/Kings/'
batch_size = 20
num_epochs = 2

def create_dummy_ds(num_samples = 100, d_img_rows = 224, d_img_cols = 224, d_img_channels= 3):
    # Creates a dummy ds for quick evaluation
    train_imgs =    np.random.random((num_samples, d_img_rows, d_img_cols, d_img_channels)) # 100 images r,c, channels
    train_pose_tx = np.random.random((num_samples, 3)) # tx, ty, tz
    train_pose_rt = np.random.random((num_samples, 4)) # w, p, q, r quarternion
    test_imgs =     np.random.random((num_samples, d_img_rows, d_img_cols, d_img_channels))
    test_pose_tx =  np.random.random((num_samples, 3)) # tx, ty,
    test_pose_rt=   np.random.random((num_samples, 4)) # w, p, q
    return(train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,
    test_pose_rt)
#num_samples = 20
def process_dataset(filename, base_dir):
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
        img_list.append(os.path.join(base_dir, filename.split('/')[-2], line_splits[0]))
        l_pose = []
        for j in range(1, len(line_splits)):
            l_pose.append(line_splits[j])
        np_pose = np.asarray(l_pose)
        pose[i,:] = np_pose
    #print(img_list[0])
    #print(pose[0,:])
    return(img_list, pose)

def read_images(img_list, img_rows, img_cols, img_channels):
    decision = False
    imgs = np.zeros((len(img_list), img_rows, img_cols, img_channels))
    for i in range(0, len(img_list)):
        im = Image.open(img_list[i])
        if(decision == True):
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
        else:
            im_r = im.resize((img_cols, img_rows), Image.BILINEAR)
            np_im = np.asarray(im_r, dtype='float32')
            imgs[i, :, :, :] = np_im
    return(imgs)

def gather_train_test_txt_list(base_dir):
    search_path = base_dir + '*/*.txt'
    txt_list = glob.glob(search_path)
    train_txt_list = []
    test_txt_list = []
    for i in range(0, len(txt_list)):
        if(txt_list[i].split('_')[-1]=='mod.txt'):
            #print(os.path.join(base_dir, txt_list[i]))
            if(txt_list[i].split('_')[-2]=='train'):
                train_txt_list.append(os.path.join(base_dir, txt_list[i]))
            elif(txt_list[i].split('_')[-2]=='test'):
                test_txt_list.append(os.path.join(base_dir, txt_list[i]))
    print(len(train_txt_list))
    return (train_txt_list, test_txt_list)

def read_image_and_pose(txt_list, img_rows, img_cols, img_channels, base_dir):
    for i in range(0, len(txt_list)):
        c_img_list, c_pose = process_dataset(txt_list[i], base_dir)
        print('Images in current list: ', len(c_img_list))
        c_imgs = read_images(c_img_list,  img_rows, img_cols, img_channels)
        if(i == 0):
            pose = c_pose
            images = c_imgs
        else:
            pose = np.concatenate((pose, c_pose), axis=0)
            images = np.concatenate((images, c_imgs), axis=0)
    print('total number samples: ', images.shape[0])
    return(images, pose)

def test_only_splits(base_dir, img_rows, img_cols, img_channels):
    train_txt_list, test_txt_list = gather_train_test_txt_list(base_dir)
    test_imgs, test_pose = read_image_and_pose(test_txt_list, img_rows, img_cols, img_channels, base_dir)
    test_imgs = test_imgs.astype('float32')
    test_pose = test_pose.astype('float32')
    test_pose_tx = test_pose[:,:3]
    test_pose_rt = test_pose[:,3:]
    test_imgs /=255
    return(test_imgs, test_pose_tx, test_pose_rt)

def load_train_test_splits(base_dir, img_rows, img_cols, img_channels):
    train_txt_list, test_txt_list = gather_train_test_txt_list(base_dir)
    train_imgs, train_pose = read_image_and_pose(train_txt_list, img_rows, img_cols, img_channels, base_dir)
    test_imgs, test_pose = read_image_and_pose(test_txt_list, img_rows, img_cols, img_channels, base_dir)
    # Preprocess the data and return the final arrays
    train_imgs = train_imgs.astype('float32')
    train_pose = train_pose.astype('float32')
    train_pose_tx = train_pose[:,:3]
    train_pose_rt = train_pose[:,3:]

    test_imgs = test_imgs.astype('float32')
    test_pose = test_pose.astype('float32')
    test_pose_tx = test_pose[:,:3]
    test_pose_rt = test_pose[:,3:]

    train_imgs /=255
    test_imgs /=255

    return(train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,
    test_pose_rt)

'''LEGACY function callbacks
def process_dataset(filename):
    with open(filename) as f:
        lines = f.readlines()
    # ============================
    #num_samples = len(lines)
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
    return(imgs)'''

def main():
    ''' LEGACY PREPROCESSING
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

    tr_tx =training_pose[:,:3]
    tr_rt =training_pose[:,3:]
    ts_tx =testing_pose[:,:3]
    ts_rt =testing_pose[:,3:]'''

    #train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx, test_pose_rt = load_train_test_splits(base_dir )

    # import the resnet model
    base_model = ResNet50(weights='imagenet')
    x = base_model.output
    x = Dropout(0.7)(x)
    x = Dense(1024, activation='tanh')(x)
    position = Dense(3, activation='tanh', name='translation')(x)
    rotation = Dense(4, activation='tanh', name='rotation')(x)

    model = Model(input=base_model.input, output=[position, rotation])
    print(model.summary())

    '''sgd = optimizers.SGD(lr = 0.00001, decay=1e-6, momentum=0.9, nesterov = True)
    model.compile(optimizer='SGD', loss='mse', loss_weights =[1.0, 500.0])

    tb = TensorBoard(log_dir='./sample_log', histogram_freq=1,
    write_graph=True, write_images=True)

    model.fit(train_imgs, [train_pose_tx, train_pose_rt], batch_size = batch_size,
    epochs= num_epochs, callbacks= [tb], shuffle=True)

    print(test_imgs.shape)
    print(test_imgs.dtype)

    ts_pos_pred, ts_rot_pred = model.predict(test_imgs, verbose=1)

    print(ts_pos_pred)
    print('==============================')
    print(ts_rot_pred)
    model.save('resnet50_pose.h5')
    #gc.collect()'''
if __name__ == '__main__':
    main()
