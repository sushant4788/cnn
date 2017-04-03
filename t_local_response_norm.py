'''test the functionality of the local reponse norm layer. the code is obtained from
https://github.com/fchollet/keras/blob/97174dd298cf4b5be459e79b0181a124650d9148/keras/layers/normalization.py#L66
'''
from __future__ import print_function
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, Input
import tensorflow as tf 
import keras_pose_resnet
from Local_Resp_Norm import LRN2D
use_dummy_ds = True
img_rows, img_cols, img_channels = 256, 455, 3
num_samples = 40
batch_size = 20
num_epochs = 5
def main():
    # Build dummy net
    train_imgs, train_pose_tx, train_pose_rt, test_imgs,test_pose_tx,test_pose_rt=keras_pose_resnet.create_dummy_ds(num_samples, img_rows,
    img_cols, img_channels)
    with tf.device('cpu:0'):
        img_input = Input(shape=(img_rows, img_cols, img_channels))
        x = Conv2D(64, (5,5), activation ='relu')(img_input)
        x = LRN2D()(x)
        x = Flatten()(x)
        y = Dense(3)(x)
        model = Model(inputs=img_input, outputs = y)
        model.compile(optimizer = 'rmsprop', loss ='mse')
        model.fit(train_imgs, train_pose_tx, batch_size =batch_size,
        epochs = num_epochs)


if __name__ == '__main__':
    main()
