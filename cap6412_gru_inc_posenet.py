'''Pose regression net that uses the Siamese features in an inception like
net and has GRU layers to regress the pose'''
from __future__ import print_function
import numpy as np
import gc, h5py, os, math, datetime, glob, transforms3d

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

def create_KITTI_dataset():
    '''Dataset is the KITTI dataset. The data is organized as follows
    '''
    ds_dir='/home/sushant/dataset_kitti/sequences/'
    ps_dir = '/home/sushant/dataset_kitti/dataset/poses/'
    #ds_search_path = os.join.path(ds_dir, '*/*.txt' )
    ps_search_path = os.path.join(ps_dir, '*.txt' )
    ps_search_txt_list = sorted(glob.glob(ps_search_path))
    print('Reading ....')
    for i in range(0, len(ps_search_txt_list)):
        print(ps_search_txt_list[i])
        c_r_R, c_r_T = read_R_and_T_from_current_seq(ps_search_txt_list[i])
        if(i == 0):
            r_R = c_r_R
            r_T = c_r_T
        else:
            r_R = np.concatenate((r_R, c_r_R), axis =0)
            r_T = np.concatenate((r_T, c_r_T), axis =0)


def main():
    '''Preprocess the dataset first'''
    create_KITTI_dataset()

if __name__ == '__main__':
    main()
