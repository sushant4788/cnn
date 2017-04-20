from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
import keras.backend as K
import tensorflow as tf
import numpy as np
import glob, os
from PIL import Image

base_dir = '/home/sushant/Downloads/Kings/'
img_rows ,img_cols, img_channels = 32, 32, 3

def process_dataset(filename):
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

def read_images(img_list):
    imgs = np.zeros((len(img_list), img_rows, img_cols, img_channels))
    for i in range(0, len(img_list)):
    #for i in range(0, 1):
        print(img_list[i])
        #loc = dataset_location + img_list[i]
        #print(loc)
        im = Image.open(img_list[i])
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


def gather_train_test_txt_list(base_dir):
    search_path = base_dir + '*/*.txt'
    txt_list = glob.glob(search_path)
    train_txt_list = []
    test_txt_list = []
    for i in range(0, len(txt_list)):
        if(txt_list[i].split('_')[-1]=='mod.txt'):
            print(os.path.join(base_dir, txt_list[i]))
            if(txt_list[i].split('_')[-2]=='train'):
                train_txt_list.append(os.path.join(base_dir, txt_list[i]))
            elif(txt_list[i].split('_')[-2]=='test'):
                test_txt_list.append(os.path.join(base_dir, txt_list[i]))
    print(train_txt_list)
    return (train_txt_list, test_txt_list)
def read_image_and_pose(txt_list):
    for i in range(0, len(txt_list)):
        c_img_list, c_pose = process_dataset(txt_list[i])
        c_imgs = read_images(c_img_list)
        if(i == 0):
            pose = c_pose
            images = c_imgs
        else:
            pose = np.concatenate((pose, c_pose), axis=0)
            images = np.concatenate((images, c_imgs), axis=0)
    return(images, pose)
def load_train_test_splits(base_dir):
    train_txt_list, test_txt_list = gather_train_test_txt_list(base_dir)
    train_imgs, train_pose = read_image_and_pose(train_txt_list)
    test_imgs, test_pose = read_image_and_pose(test_txt_list)
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
def main():
    train_imgs, train_pose_tx, train_pose_rt, test_imgs, test_pose_tx,test_pose_rt = load_train_test_splits(base_dir )
if __name__ == '__main__':
    main()


'''t = K.ones((12, 3))
t1 = t[:, :1] + 1
t2 = t[:, 1:] - 1
t3 = K.concatenate([t1, t2])
print(K.eval(t3))

def mean_squared_error(y_true, y_pred):
    err = y_pred - y_true
    print('difference is : \n', K.eval(err))
    sq_err = K.square(err)
    print('square err: \n', K.eval(sq_err))
    mse_ = K.mean(K.square(y_pred - y_true), axis=-1)
    print('mse: \n', K.eval(mse_))
    return K.mean(K.square(y_pred - y_true), axis=-1)
print('============================================')
y_true = K.random_normal((1,2,2))
print('y_true: \n ', K.eval(y_true))
y_pred = K.random_normal((1,2,2))
print('y_pred: \n', K.eval(y_pred))
loss = mean_squared_error(y_true, y_pred)

print(K.eval(loss))'''

#model = ResNet50(weights='imagenet')
#print('Model summary: \n', model.summary())


'''def custom_loss(y_pred, y_truth):
    # ============================================================
    # Define the mean squared error :
    def mean_squared_error(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
    # ============================================================
    beta_val = 100
    # ============================================================
    # seperate the first 3 elems
    pos_pred = y_pred[:3]
    print('pos pred ', K.eval(pos_pred))
    rot_pred = y_pred[3:]
    pos_truth = y_truth[:3]
    rot_truth = y_truth[3:]
    beta = tf.constant(beta_val, dtype='float32',name='beta')
    pos_loss = mean_squared_error(pos_truth, pos_pred)
    print('pos_loss : \n', K.eval(pos_loss))
    print(pos_loss)
    rot_loss = mean_squared_error(rot_truth, rot_pred)
    print(rot_loss)
    print('rot_loss : \n', K.eval(rot_loss))
    print(K.eval(beta*rot_loss))
    return(pos_loss + beta*rot_loss)

#y_true = K.random_normal((7,))
#y_pred = K.random_normal((7,))
y_true = tf.constant([1,0,1,0,1,1,1], dtype='float32')
y_pred = tf.constant([1,20,1,0,0,0,1], dtype='float32')
loss = custom_loss(y_pred, y_true)
print('custom loss : ', K.eval(loss))'''

'''t = K.ones((7, ))
t1 = t[:3] + 1
t2 = t[3:] - 1
t3 = K.concatenate([t1, t2])
t4 = tf.norm(t3)
t5 = t3/t4
print(K.eval(t3))
print(K.eval(t4))
print(K.eval(t5))'''
