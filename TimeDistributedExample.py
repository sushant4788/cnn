from __future__ import print_function
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.optimizers import RMSprop, Adadelta
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed

np.random.seed(2016)
K.set_image_dim_ordering ='tf'
batch_size      = 32
nb_epochs       = 20
examplesPer     = 60000
maxToAdd        = 8
hidden_units    = 200
size            = 28
(X_train_raw, y_train_temp), (X_test_raw, y_test_temp) = mnist.load_data()
# basic processing
X_train_raw = X_train_raw.astype('float32')
X_test_raw  = X_test_raw.astype('float32')
X_train_raw /= 255
X_test_raw  /= 255

#X_test_raw = np.expand_dims(X_test_raw, axis=3)
print('Building model')
img_rows, img_cols, img_channels = 28, 28, 1
'''model = Sequential()
model.add(Conv2D(64, (7,7), activation='relu', input_shape()))
model.add(Conv2D(32, (5,5), activation = 'relu'))
model.add(Flatten())
model.add(GRU(output_dim=100, return_sequences))
'''
