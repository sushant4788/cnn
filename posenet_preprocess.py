from __future__ import print_function
from PIL import Image
import numpy as np
import gc, glob, os, h5py, random
#===========================================================================
'''Resize the image to approximately half the size'''
RESIZE_ROWS = 256
RESIZE_COLS = 455
#===========================================================================
def create_dummy_ds(num_samples = 100, d_img_rows = 224, d_img_cols = 224, d_img_channels= 3):
    '''# Creates a dummy ds for quickimg_list evaluation.'''

    train_imgs =    np.random.random((num_samples, d_img_rows, d_img_cols, d_img_channels)) # 100 images r,c, channels
    train_pose_tx = np.random.random((num_samples, 3)) # tx, ty, tz
    train_pose_rt = np.random.random((num_samples, 4)) # w, p, q, r quarternion
    test_imgs =     np.random.random((num_samples, d_img_rows, d_img_cols, d_img_channels))
    test_pose_tx =  np.random.random((num_samples, 3)) # tx, ty,
    test_pose_rt=   np.random.random((num_samples, 4)) # w, p, q
    return(train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,
    test_pose_rt)

def test_only_splits(base_dir, img_rows, img_cols, img_channels):
    '''Generate the test only splits. Used incase the model is already trained
    and only needs to be tested'''
    train_txt_list, test_txt_list = gather_train_test_txt_list(base_dir)
    test_imgs, test_pose = read_image_and_pose(test_txt_list, img_rows, img_cols, img_channels, base_dir)
    test_imgs = test_imgs.astype('float32')
    test_pose = test_pose.astype('float32')
    test_pose_tx = test_pose[:,:3]
    test_pose_rt = test_pose[:,3:]
    #test_imgs /=255
    return(test_imgs, test_pose_tx, test_pose_rt)

def read_pose_val_and_img_list(filename, base_dir):
    '''From the file, read the pose values and the list of images '''
    with open(filename) as f:
        lines = f.readlines()
    num_samples = len(lines)
    lines = [x.strip('\n') for x in lines]
    pose = np.zeros((num_samples, 7), dtype='float32')
    img_list = []
    for i in range(0, num_samples):
        line_splits = lines[i].split(' ')
        img_list.append(os.path.join(base_dir, filename.split('/')[-2], line_splits[0]))
        l_pose = []
        for j in range(1, len(line_splits)):
            l_pose.append(line_splits[j])
        np_pose = np.asarray(l_pose)
        pose[i,:] = np_pose
    return(img_list, pose)

def read_training_images(img_list, img_rows, img_cols, img_channels):
    '''Read the images from the list of the training images, resize the images
    to 455 x 256 as done in the PoseNet paper and then randomly crop a path of
    size 224 x 224 from this image.'''
    imgs = np.zeros((len(img_list), img_rows, img_cols, img_channels), dtype='float32')
    for i in range(0, len(img_list)):
        im = Image.open(img_list[i])
        # Resize the image
        im_r = im.resize((RESIZE_COLS, RESIZE_ROWS), Image.BILINEAR)
        np_im = np.asarray(im_r, dtype='float32')
        s_x = random.randint(0,230)
        s_y = random.randint(0,30)
        #print('s_x', s_x)
        #print('s_y', s_y)
        c_im = np_im[s_y: s_y+img_rows, s_x:s_x+img_cols,:]
        #print(c_im.shape)
        imgs[i,:,:,:] = c_im
    return(imgs)

def read_testing_images(img_list, img_rows, img_cols, img_channels):
    '''Read the images from the list of tesing images, resize the images to
    455 x 256 as done in the PoseNet paper then crop a path of size 224 x 224
    from the center of the image'''
    imgs = np.zeros((len(img_list), img_rows, img_cols, img_channels), dtype ='float32')
    for i in range(0, len(img_list)):
        im = Image.open(img_list[i])
        im_r = im.resize((RESIZE_COLS, RESIZE_ROWS), Image.BILINEAR)
        np_im = np.asarray(im_r, dtype='float32')
        y, x, c = np_im.shape
        c_y = np.floor(y/2)
        c_x = np.floor(x/2)
        s_y = int(np.floor(c_y - (img_rows/2)))
        s_x = int(np.floor(c_x - (img_cols/2)))
        c_im = np_im[s_y: s_y+img_rows, s_x: s_x+ img_cols, :]
        imgs[i, :,:,:] = c_im
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
    #print(len(train_txt_list))
    return (train_txt_list, test_txt_list)

def read_image_and_pose(txt_list, img_rows, img_cols, img_channels, base_dir, is_train):
    for i in range(0, len(txt_list)):
        c_img_list, c_pose = read_pose_val_and_img_list(txt_list[i], base_dir)
        print('Images in current list: ', len(c_img_list))
        if(is_train == True):
            c_imgs = read_training_images(c_img_list, img_rows, img_cols, img_channels)
        else:
            c_imgs = read_testing_images(c_img_list, img_rows, img_cols, img_channels)
        if(i == 0):
            pose = c_pose
            images = c_imgs
        else:
            pose = np.concatenate((pose, c_pose), axis=0)
            images = np.concatenate((images, c_imgs), axis=0)

    return(images, pose)

def load_train_test_splits(base_dir, img_rows, img_cols, img_channels):
    train_txt_list, test_txt_list = gather_train_test_txt_list(base_dir)
    train_imgs, train_pose = read_image_and_pose(train_txt_list, img_rows, img_cols, img_channels, base_dir, True)
    test_imgs, test_pose = read_image_and_pose(test_txt_list, img_rows, img_cols, img_channels, base_dir, False)
    # Preprocess the data and return the final arrays
    train_pose_tx = train_pose[:,:3]
    train_pose_rt = train_pose[:,3:]

    test_pose_tx = test_pose[:,:3]
    test_pose_rt = test_pose[:,3:]

    return(train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,
    test_pose_rt)


def main():
    # Set some hardcoded sizes
    img_rows ,img_cols, img_channels = 224, 224, 3
    # Some dir locations
    # training_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_train_mod.txt'
    # testing_file = '/home/sushant/Downloads/Kings/KingsCollege/dataset_test_mod.txt'
    # dataset_location ='/home/sushant/Downloads/Kings/KingsCollege/'
    base_dir ='/home/sushant/Downloads/Kings/'
    base_dir ='/home/sushant/Downloads/'
    train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx, test_pose_rt = load_train_test_splits(base_dir, img_rows, img_cols, img_channels )
    save_prefix = '/home/sushant/dataset_mod/'
    train_path = os.path.join(save_prefix, 'train.h5')
    test_path = os.path.join(save_prefix, 'test.h5')

    h5f = h5py.File(train_path , 'w')
    h5f.create_dataset('train_imgs', data=train_imgs, dtype='float32')
    h5f.create_dataset('train_pose_tx', data=train_pose_tx, dtype='float32')
    h5f.create_dataset('train_pose_rt', data=train_pose_rt, dtype='float32')
    h5f.close()

    h5f = h5py.File(test_path, 'w')
    h5f.create_dataset('test_imgs', data=test_imgs, dtype='float32')
    h5f.create_dataset('test_pose_tx', data=test_pose_tx, dtype='float32')
    h5f.create_dataset('test_pose_rt', data=test_pose_rt, dtype='float32')
    h5f.close()

    #h5f = h5py.File('train.h5','r')
    #b = h5f['train_imgs'][:]
    #c = h5f['train_pose_tx'][:]
    #d = h5f['train_pose_rt'][:]
    #print(b.shape)
    #print(c.shape)
    #print(d.shape)
    #h5f.close()

if __name__ == '__main__':
    main()


'''def read_images(img_list, img_rows, img_cols, img_channels):
    #Legacy function to read images. Not used any more
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
            imgs[i, :, :, :] = c_im
        else:
            im_r = im.resize((img_cols, img_rows), Image.BILINEAR)
            np_im = np.asarray(im_r, dtype='float32')
            imgs[i, :, :, :] = np_im
    return(imgs) '''
