from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions


import keras.backend as K
import tensorflow as tf
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


def custom_loss(y_pred, y_truth):
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
print('custom loss : ', K.eval(loss))

'''t = K.ones((7, ))
t1 = t[:3] + 1
t2 = t[3:] - 1
t3 = K.concatenate([t1, t2])
t4 = tf.norm(t3)
t5 = t3/t4
print(K.eval(t3))
print(K.eval(t4))
print(K.eval(t5))'''
