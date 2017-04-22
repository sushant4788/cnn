'''Load the original train and the test data and split it according to the
datasets seperately'''
from __future__ import print_function
import h5py
import numpy as np

def main():
    write_dir = 'landmarks_seperate/'
    ds_dir = '/home/sushant/dataset_mod/'
    train_file_name = ds_dir+'train.h5'
    test_file_name = ds_dir+ 'test.h5'
    # name of the locations
    location_list = ['OldHospital', 'StMarysChurch', 'KingsCollege', 'Street', 'ShopFacade']
    tr_locs = [0, 895, 2382, 3602, 6617, 6848]
    ts_loc = [0, 182, 712, 1055, 3978, 4081]
    # Load the complete train and the testing set
    train = h5py.File(train_file_name, 'r')
    train_imgs = train['train_imgs'][:]
    train_pose_tx = train['train_pose_tx'][:]
    train_pose_rt = train['train_pose_rt'][:]
    print('TRAIN set has ', train_imgs.shape[0], ' number of images')

    train.close()
    test = h5py.File(test_file_name, 'r')
    test_imgs = test['test_imgs'][:]
    test_pose_tx = test['test_pose_tx'][:]
    test_pose_rt = test['test_pose_rt'][:]
    print('TEST set has ', test_imgs.shape[0], ' number of images')

    test.close()
    for i in range(0, len(location_list)):
        print(location_list[i])
        ds_tr_imgs = train_imgs[ts_loc[i]:ts_loc[i+1],:,:,:]
        ds_tr_pose_tx = train_pose_tx[ts_loc[i]:ts_loc[i+1],:]
        ds_tr_pose_rt = train_pose_rt[ts_loc[i]:ts_loc[i+1],:]
        print('current train set has ', ds_tr_imgs.shape[0], ' number of images')
        ds_ts_imgs = test_imgs[tr_loc[i]:tr_loc[i+1], :,:,:]
        ds_ts_pose_tx = test_pose_tx[tr_loc[i]:tr_loc[i+1],:]
        ds_ts_pose_rt = test_pose_rt[tr_loc[i]:tr_loc[i+1],:]
        print('current test set has ', ds_ts_imgs.shape[0], ' number of images')
        # save the training dataset
        h5f = h5py.File(write_dir+location_list[i]+'.h5', 'w')
        h5f.create_dataset('train_imgs', data=ds_tr_imgs, dtype='float32')
        h5f.create_dataset('train_pose_tx', data=ds_tr_pose_tx, dtype='float32')
        h5f.create_dataset('train_pose_rt', data=ds_tr_pose_rt, dtype='float32')
        h5f.create_dataset('test_imgs', data=ds_ts_imgs, dtype='float32')
        h5f.create_dataset('test_pose_tx', data=ds_ts_pose_tx, dtype='float32')
        h5f.create_dataset('test_pose_rt', data=ds_ts_pose_rt, dtype='float32')
        h5f.close()

if __name__ == '__main__':
    main()
