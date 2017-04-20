from __future__ import print_function
import cap6412_resnet_inception
import numpy as np
import h5py
import keras.backend as K
import tensorflow as tf
import math
from keras.callbacks import TensorBoard

use_dummy_ds = False
use_gpu = True
batch_size = 75
num_epochs = 500

train_splits = [0, 895, 1487, 1220, 3015, 1487, 1532]
test_splits = [0, 182, 530, 343, 2923, 103, 760]

def main():
    img_rows, img_cols, img_channels = 224, 224, 3

    ds_dir = '/home/sushant/dataset_mod/'
    train_prefix = 'train.h5'
    test_prefix = 'test.h5'

    train = h5py.File(ds_dir+train_prefix, 'r')
    train_imgs = train['train_imgs'][:]
    train_pose_tx = train['train_pose_tx'][:]
    train_pose_rt = train['train_pose_rt'][:]
    train.close()

    test = h5py.File(ds_dir+test_prefix, 'r')
    test_imgs = test['test_imgs'][:]
    test_pose_tx = test['test_pose_tx'][:]
    test_pose_rt = test['test_pose_rt'][:]
    test.close()
    median_result = np.zeros((len(train_splits)-1, 2), dtype='float32')

    for k in range(0, len(train_splits)-1):
        # Get the current training data
        c_train_imgs = train_imgs[train_splits[k]:train_splits[k] + train_splits[k+1],:,:,:]
        c_train_pose_tx = train_pose_tx[train_splits[k]:train_splits[k] + train_splits[k+1],:]
        c_train_pose_rt = train_pose_rt[train_splits[k]:train_splits[k] + train_splits[k+1],:]
        c_test_imgs = test_imgs[test_splits[k]:test_splits[k] + test_splits[k+1], :,:,:]
        c_test_pose_tx = test_pose_tx[test_splits[k]:test_splits[k] + test_splits[k+1],:]
        c_test_pose_rt = test_pose_rt[test_splits[k]:test_splits[k] + test_splits[k+1],:]
        # Load the current model
        tb_log_dir_name  = './consoliated_logs'
        model_name = 'ds_' + str(k+1) + '.h5'
        with tf.device('gpu:0'):
            model = cap6412_resnet_inception.inc_pose_net(img_rows, img_cols, img_channels)
            tb = TensorBoard(log_dir=tb_log_dir_name, histogram_freq=1,
            write_graph=True, write_images=True)
            model.fit(c_train_imgs, [c_train_pose_tx, c_train_pose_rt, c_train_pose_tx, c_train_pose_rt,c_train_pose_tx, c_train_pose_rt],
            batch_size= batch_size, callbacks = [tb], epochs = num_epochs, shuffle=False)
            model.save(model_name)
            p_tx_1, p_rx_1, p_tx_2, p_rx_2, p_tx_3, p_rx_3 = model.predict(c_test_imgs)

            results = np.zeros((c_test_imgs.shape[0], 2), dtype = 'float32')
            for i in range(0, c_test_imgs.shape[0]):
                q2 = p_rx_3[i,:] / np.linalg.norm(p_rx_3[i,:])
                q1 = c_test_pose_rt[i, :] / np.linalg.norm(c_test_pose_rt[i,:])
                d = abs(np.sum(np.multiply(q1, q2)))
                theta = 2*np.arccos(d) * 180/math.pi
                error_x = np.linalg.norm(c_test_pose_tx[i, :] - p_tx_3[i, :])
                results[i, :] = [error_x, theta]
                print ('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
            median_result[k,:] = np.median(results,axis=0)
            print('Median error meters: ', median_result[k,0])
            print('Median error degrees: ', median_result[k,1])
            save_dir_name = 'results_' + str(i) + '.txt'
            np.savetxt(save_dir_name, results, delimiter=' ')
            print( 'Success!')
            K.clear_session()



if __name__ == '__main__':
    main()
