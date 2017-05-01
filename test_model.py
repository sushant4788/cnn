'''Load the posenet model and test the performance'''
from __future__ import print_function
import numpy as np
import warnings
from keras import layers
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import load_model
import gc, h5py, datetime
import stereo_regression_gru 

def main():
    # Some params 
    sequence_length, img_rows, img_cols, img_channels = 7, 224, 224, 1
    # Load the dataset
    dataset_dir = '/home/sushant/kitti_dataset/'
    train_imgs, train_p_imgs, train_pose_rt, train_pose_tx, test_imgs, test_p_imgs, test_pose_rt, test_pose_tx = stereo_regression_gru.read_KITTI_ds(dataset_dir)
    test_imgs = test_imgs[:3297, :,:,:]
    test_p_imgs = test_p_imgs[:3297, :,:,:]
    test_pose_rt = test_pose_rt[:3297,:]
    test_pose_tx = test_pose_tx[:3297,:]
    print(test_imgs.shape[0])
    # Some preporcessing
    test_imgs -= train_imgs.mean()
    # Reshape the array 
    test_imgs = np.reshape(test_imgs, (-1, sequence_length, img_rows, img_cols, img_channels))
    test_p_imgs = np.reshape(test_p_imgs, (-1, sequence_length, img_rows, img_cols, img_channels))
    test_pose_rt = np.reshape(test_pose_rt, (-1, sequence_length, 4))
    test_pose_tx = np.reshape(test_pose_tx, (-1, sequence_length, 3))
    #Model file name
    model_file = 'stereo_reg_gru2017-04-30-16-53-00-820403.h5'
    # Load the model
    model = load_model(model_file)
    # Get the prodictions
    p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict([test_imgs, test_p_imgs])
    # get the errors
    stereo_regression_gru.get_pose_errors(p_tx_3, p_rx_3, test_pose_tx, test_pose_rt)

if __name__ == '__main__':
    main()
