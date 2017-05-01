'''Load the posenet model and test the performance'''
from __future__ import print_function
import numpy as np
import warnings
from keras import layers
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import load_model
import gc, h5py, datetime
from stereo_reg_gru import get_pose_errors, read_KITTI_ds

def main():
    # Load the dataset
    dataset_dir = '/home/sushant/kitti_dataset/'
    train_imgs, train_p_imgs, train_pose_rt, train_pose_tx, test_imgs, test_p_imgs, test_pose_rt, test_pose_tx = read_KITTI_ds(dataset_dir)
    # Some preporcessing
    test_imgs -= train_imgs.mean()
    #Model file name
    model_file = 'stereo_reg_gru2017-04-30-16-53-00-820403.h5'
    # Load the model
    model = load_model(model_file)
    # Get the prodictions
    p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict()
    # get the errors
    get_pose_errors(p_tx_3, p_rx_3, test_pose_tx, test_pose_rt)

if __name__ == '__main__':
    main()
