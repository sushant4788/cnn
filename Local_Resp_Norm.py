from keras.layers.core import Layer
import tensorflow as tf
class LRN2D(Layer):
    """
    Modified from :
    https://github.com/fchollet/keras/blob/97174dd298cf4b5be459e79b0181a124650d9148/keras/layers/normalization.py#L66
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt

    """

    def __init__(self, alpha=1e-4, k=1, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        '''Making changes to allow it to run on tensorflow '''
        X = self.get_input(train)
        b, r, c, ch = X.shape
        half_n = self.n // 2
        input_sqr = tf.square(X)
        extra_channels = tf.zeros([b, r, c, ch + 2*half_n], dtype=tf.float32)
        input_sqr = tf.assign(input_sqr, extra_channels[:,:,:,ch + 2*half_n])
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, :, :, i:i+ch]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}
