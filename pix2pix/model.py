
from tensorflow.keras.layers import Add, Conv2D, Input, ReLU, Layer, Lambda, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, Dense, Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D, Concatenate, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

IMAGE_SIZE = 512
GLOBAL_M = 32

kernel_initializer = 'random_normal'


def encode_layer(x, k = 4, mult = 1, drop = False, bn = True):
    x = Conv2D(filters = GLOBAL_M*mult, 
               kernel_size = k,
               strides = 2,
               padding = 'same',
               activation = None,
               kernel_initializer=kernel_initializer,
               use_bias = False)(x)        
    #if bn: x = BatchNormalization()(x)
    if bn: x = InstanceNormalization()(x)
    if drop: x = Dropout(0.5)(x)
    x = LeakyReLU()(x)
    return x

class PixelShuffle(Layer):
    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.scale = scale
    def call(self, inputs):
        inputs = tf.nn.depth_to_space(inputs,self.scale)
        return inputs


def UpSampling2DPS(x, k = 4, bn = True):
    x = PixelShuffle(scale = 2)(x)
    return x


def decode_layer(x, y, k = 3, mult = 1, last_layer = False, drop = False, bn = True):
    #x  = UpSampling2D(size=2)(x)   
    x = UpSampling2DPS(x)
    x = Concatenate()([x,y])
    if last_layer:
        x = Conv2D(filters = 3, 
                   kernel_size = k,
                   strides = 1, 
                   padding = 'same',
                   activation = 'tanh',
                   use_bias = False,
                   kernel_initializer = kernel_initializer)(x)
        #x = ReLU(max_value = 255.)(x)
    else:
        x = Conv2D(filters = GLOBAL_M*mult,
                   kernel_size = k, 
                   strides = 1,
                   padding = 'same',
                   activation = None,
                   use_bias = False,
                   kernel_initializer = kernel_initializer)(x)
        #if bn: x = BatchNormalization()(x)
        if bn: x = InstanceNormalization()(x)
        if drop: x = Dropout(0.5)(x)
        #x = ReLU()(x)
        x = LeakyReLU()(x)
    return x


def crt_D():
    
    inp = Input((IMAGE_SIZE,IMAGE_SIZE,3), name='input_image')
    tar = Input((IMAGE_SIZE,IMAGE_SIZE,3), name='target_image')
    x = Concatenate()([inp, tar])
   
    x = encode_layer(x = x, k = 4, mult = 1, bn = False)
    x = encode_layer(x = x, k = 4, mult = 2)
    x = encode_layer(x = x, k = 4, mult = 4)
    x = encode_layer(x = x, k = 4, mult = 8)
    x = ZeroPadding2D()(x)  # (bs, 34, 34, 256)
    x = Conv2D(filters = 512, # (bs, 31, 31, 512)
                   kernel_size = 4, 
                   strides = 1, 
                   padding = 'valid',
                   activation = None,
                   use_bias = False,
                   kernel_initializer = kernel_initializer)(x)
    #x = BatchNormalization()(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU()(x)
    x = ZeroPadding2D()(x)  # (bs, 33, 33, 512)
    x = Conv2D(filters = 1, # (bs, 30, 30, 1)
                   kernel_size = 4, 
                   strides = 1, 
                   padding = 'valid',
                   activation = None,
                   use_bias = False,
                   kernel_initializer = kernel_initializer)(x)
    m = Model(inputs = [inp,tar], outputs = x)
    return m
#m = crt_D()
#print(m.summary())


def crt_G():
    inputs = Input((IMAGE_SIZE,IMAGE_SIZE,3))

    x1 = encode_layer(x = inputs, k = 4, mult = 1, bn = False)
    x2 = encode_layer(x = x1, k = 4, mult = 2)
    x3 = encode_layer(x = x2, k = 4, mult = 4)
    x4 = encode_layer(x = x3, k = 4, mult = 8)
    x5 = encode_layer(x = x4, k = 4, mult = 8)
    x6 = encode_layer(x = x5, k = 4, mult = 8)
    x7 = encode_layer(x = x6, k = 4, mult = 8,drop=True)
    x8 = encode_layer(x = x7, k = 4, mult = 8,drop=True)
 
    x9 = decode_layer(x = x8, y = x7, k = 4, mult = 8)
    x10 = decode_layer(x = x9, y = x6, k = 4, mult = 8)
    x11 = decode_layer(x = x10, y = x5, k = 4, mult = 8)
    x12 = decode_layer(x = x11, y = x4, k = 4, mult = 8)
    x13 = decode_layer(x = x12, y = x3, k = 4, mult = 4)
    x14 = decode_layer(x = x13, y = x2, k = 4, mult = 2)
    x15 = decode_layer(x = x14, y = x1, k = 4, mult = 1)
    out = decode_layer(x = x15, y = inputs, k = 4, last_layer=True)
    m = Model(inputs = inputs, outputs = out)
    return m
