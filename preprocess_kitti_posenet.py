'''Pose regression net that uses the Siamese features in an inception like
net and has GRU layers to regress the pose'''
from __future__ import print_function
from PIL import Image
import numpy as np
import gc, h5py, os, math, datetime, glob, transforms3d, random
RESIZE_COLS, RESIZE_ROWS = 828, 250

def relative_pose(R_p, T_p, R_c, T_c):
    '''Get the realtive pose in between the two matrices'''
    delta_T = T_p - T_c
    delta_R = np.dot(R_p, R_c)
    return(delta_R, delta_T)

def read_R_and_T_from_current_seq(sequence_name):
    with open(sequence_name) as f:
        lines = f.readlines()
    lines = [x.strip('\n') for x in lines]
    num_samples = len(lines)
    # pose = np.zeros((num_samples, 7), dtype = 'float32')
    # Create different matrices fot the rotations and translations
    relative_rotations= np.zeros((num_samples, 4), dtype='float32')
    relative_translations = np.zeros((num_samples,3), dtype='float32')
    print('Current sequence has ', num_samples, ' frames')
    for j in range(0, num_samples):
    #for j in range(0, 1):
        l_splits = lines[j].split(' ')
        l_pose = []
        for k in range(0, len(l_splits)):
        #for k in range(0, 1):
            l_pose.append(l_splits[k])
        np_c_pose = np.asarray(l_pose, dtype='float32')
        np_c_pose = np.reshape(np_c_pose, (3,4))
        #Get the current R and T wrt to C0
        R_c = np_c_pose[0:3,0:3]
        T_c = np_c_pose[:,3]
        # Get the relative pose of the camera with respect to the previous
        # frame
        if(j == 0):
            # initialize the first frame values
            delta_R = np.eye(3, dtype='float32')
            delta_T = np.zeros((1,3), dtype='float32')
        else:
            # For every other frame
            delta_R, delta_T = relative_pose(R_p, T_p, R_c, T_c)
        #Convert the matrices into quarternions
        delta_R = transforms3d.quaternions.mat2quat(delta_R)

        relative_rotations[j, :] = delta_R
        relative_translations[j,:] = delta_T
        # The current becomes the previous pose for the next iteration
        R_p = R_c
        T_p = T_c
    return(relative_rotations, relative_translations)

def read_train_images_from_seq_dir(ds_dir, seq_dir, img_rows, img_cols, img_channels):
    #Read the training images from the dequences folder. The procedure for
    #generating the train and testing samples is slightly different

    c_seq = os.path.join(ds_dir, seq_dir, 'image_0', '*.png')
    c_seq_imgs = sorted(glob.glob(c_seq))
    num_frames = len(c_seq_imgs)
    print('current seq has :' , num_frames, ' of frames')
    imgs = np.zeros((num_frames, img_rows, img_cols, img_channels), dtype='float32')
    # previous images
    p_imgs = np.zeros((num_frames, img_rows, img_cols, img_channels), dtype='float32')
    c_im = np.zeros((img_rows, img_cols,1), dtype = 'float32')
    for j in range(0, num_frames):
        im = Image.open(c_seq_imgs[j])
        im_r = im.resize((RESIZE_COLS, RESIZE_ROWS), Image.BILINEAR)
        np_im = np.asarray(im_r, dtype='float32')
        #print(np_im.shape)
        s_x = random.randint(0,600)
        s_y = random.randint(0,24)
        #print('===', s_y, s_x)
        #print('===', s_y+img_rows, s_x+img_cols)
        c_im[:,:,0] = np_im[s_y: s_y+img_rows, s_x:s_x+img_cols]
        imgs[j,:,:,:] = c_im
        if(j == 0):
            p_imgs[j,:,:,:] = c_im
        else:
            p_imgs[j-1,:,:,:] = c_im
    return(imgs, p_imgs)


def read_test_images_from_seq_dir(ds_dir, seq_dir, img_rows, img_cols, img_channels):
    # Read the testing images from the sequences folder.
    c_seq = os.path.join(ds_dir, seq_dir, 'image_0', '*.png')
    c_seq_imgs = sorted(glob.glob(c_seq))
    num_frames = len(c_seq_imgs)
    print('current seq has :' , num_frames, ' of frames')
    imgs = np.zeros((num_frames, img_rows, img_cols, img_channels), dtype='float32')
    p_imgs = np.zeros((num_frames, img_rows, img_cols, img_channels), dtype='float32')
    c_im = np.zeros((img_rows, img_cols,1), dtype = 'float32')
    for j in range(0, num_frames):
        im = Image.open(c_seq_imgs[j])
        im_r = im.resize((RESIZE_COLS, RESIZE_ROWS), Image.BILINEAR)
        np_im = np.asarray(im_r, dtype='float32')
        y, x = np_im.shape
        c_y = np.floor(y/2)
        c_x = np.floor(x/2)
        s_y = int(np.floor(c_y - (img_rows/2)))
        s_x = int(np.floor(c_x - (img_cols/2)))
        c_im[:,:,0] = np_im[s_y: s_y+img_rows, s_x: s_x+ img_cols]
        imgs[j, :,:,:] = c_im
        if(j == 0):
            p_imgs[j,:,:,:] = c_im
        else:
             p_imgs[j-1,:,:,:] = c_im
    return(imgs, p_imgs)

def create_KITTI_dataset(img_rows, img_cols, img_channels):
    '''Dataset is the KITTI dataset. The data is organized as follows
    '''
    ds_dir='/home/sushant/dataset_kitti/sequences/'
    ps_dir = '/home/sushant/dataset_kitti/dataset/poses/'
    ps_search_path = os.path.join(ps_dir, '*.txt' )
    ps_search_txt_list = sorted(glob.glob(ps_search_path))
    '''print('Reading ....')
    for i in range(0, len(ps_search_txt_list)):
        print(ps_search_txt_list[i])
        c_r_R, c_r_T = read_R_and_T_from_current_seq(ps_search_txt_list[i])
        if(i == 0):
            r_R = c_r_R
            r_T = c_r_T
        else:
            r_R = np.concatenate((r_R, c_r_R), axis =0)
            r_T = np.concatenate((r_T, c_r_T), axis =0)'''
    # Process the images
    sorted_seq_folders =sorted(os.listdir(ds_dir))
    # We only have the ground truth for a subset of the dataset
    print('Now reading the images from the KITTI ds ')
    train_split = 8 # split at 8
    for i in range(0, len(ps_search_txt_list)):
        print('Reading', ps_search_txt_list[i])
        #print('Reading ', sorted_seq_folders[i])
        if(i < train_split):
            # Training set
            print('adding to train')
            c_imgs, p_imgs  =read_train_images_from_seq_dir(ds_dir, sorted_seq_folders[i],
            img_rows, img_cols, img_channels)
            # Subtract the mean
            # c_imgs -= c_imgs.mean()
            c_r_R, c_r_T = read_R_and_T_from_current_seq(ps_search_txt_list[i])

            if(i ==0):
                train_imgs = c_imgs
                train_p_imgs = p_imgs
                train_R = c_r_R
                train_T = c_r_T
            else:
                train_imgs = np.concatenate((train_imgs, c_imgs), axis=0)
                train_p_imgs = np.concatenate((train_p_imgs, p_imgs), axis=0)
                train_R = np.concatenate((train_R, c_r_R), axis =0)
                train_T = np.concatenate((train_T, c_r_T), axis =0)
        else:
            print('adding to test')
            c_imgs, p_imgs = read_test_images_from_seq_dir(ds_dir, sorted_seq_folders[i],
            img_rows, img_cols, img_channels)
            if(i == train_split):
                test_imgs = c_imgs
                test_p_imgs = p_imgs
                r_R = c_r_R
                r_T = c_r_T
            else:
                test_imgs = np.concatenate((test_imgs, c_imgs), axis =0)
                test_p_imgs = np.concatenate((test_p_imgs, p_imgs), axis=0)
                test_R = np.concatenate((test_R, c_r_R), axis =0)
                test_T = np.concatenate((test_T, c_r_T), axis =0)
    print('Train imgs shape: ', train_imgs.shape, ' and test imgs shape is : ', test_imgs.shape)
    return(train_imgs, train_p_imgs, train_R, train_T, test_imgs, test_p_imgs, test_R, test_T)

def main():
    '''Preprocess the dataset first'''
    img_rows, img_cols, img_channels = 224, 224, 3
    train_imgs, test_p_imgs, train_R, train_T, test_imgs, test_p_imgs, test_R, test_T = create_KITTI_dataset(img_rows, img_cols, img_channels)
    # save the data in h5 format
    h5f = h5py.File('train.h5', 'w')
    h5f.create_dataset('train_imgs', data=train_imgs, dtype='float32')
    h5f.create_dataset('train_p_imgs', data=train_p_imgs, dtype='float32')
    h5f.create_dataset('train_R', data=train_R, dtype='float32')
    h5f.create_dataset('train_T', data=train_T, dtype='float32')
    h5f.close()

    h5f = h5py.File('test.h5', 'w')
    h5f.create_dataset('test_imgs', data=test_imgs, dtype='float32')
    h5f.create_dataset('test_p_imgs', data=test_p_imgs, dtype='float32')
    h5f.create_dataset('test_R', data=test_R, dtype='float32')
    h5f.create_dataset('test_T', data=test_T, dtype='float32')
    h5f.close()
if __name__ == '__main__':
    main()
