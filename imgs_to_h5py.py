'''This function reads the images from the directory location on the disk and
saves them in contigious h5py format'''

from __future__ import print_function
from PIL import Image
import h5py, glob, os
import numpy as np
import posenet_preprocess

def read_imgs_and_pose(img_rows, img_cols, img_channels, txt_list, base_dir, is_train):
    ds_name = []
    for i in range(0, len(txt_list)):
        c_ds_name = txt_list[i].split('/')[-2]
        ds_name.append(c_ds_name)
        c_img_list, c_pose = posenet_preprocess.read_pose_val_and_img_list(txt_list[i], base_dir)
        if(is_train == True):
            c_imgs = posenet_preprocess.read_training_images(c_img_list, img_rows, img_cols, img_channels)
            prefix = 'train_'
        else:
            c_imgs = posenet_preprocess.read_testing_images(c_img_list, img_rows, img_cols, img_channels)
            prefix ='test_'
        c_imgs /=255.0
        c_imgs-=c_imgs.mean()

        pose_tx = c_pose[:,:3]
        pose_rt = c_pose[:,3:]

        filename= prefix + c_ds_name + '.h5'
        print(filename)
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset(prefix + 'imgs', data = c_imgs, dtype ='float32')
        h5f.create_dataset(prefix + 'pose_tx', data = pose_tx, dtype ='float32')
        h5f.create_dataset(prefix + 'pose_rt', data = pose_rt, dtype ='float32')
        h5f.close


def main():
    base_dir = '/home/sushant/Downloads/' # location of the ds root dir
    img_rows, img_cols, img_channels = 224, 224, 3
    train_txt_list, test_txt_list = posenet_preprocess.gather_train_test_txt_list(base_dir)
    read_imgs_and_pose(img_rows, img_cols, img_channels, train_txt_list, base_dir, True)
    read_imgs_and_pose(img_rows, img_cols, img_channels, test_txt_list, base_dir, False)

    #h5f = h5py.File('test_KingsCollege.h5','r')
    #b = h5f['test_imgs'][:]
    #c = h5f['test_pose_tx'][:]
    #d = h5f['test_pose_rt'][:]
    #print(b.shape)
    #print(c.shape)
    #print(d.shape)
    #h5f.close()
if __name__ == '__main__':
    main()
