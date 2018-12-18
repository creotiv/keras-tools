from keras import backend as K
from keras.layers import Conv2D, Input, Activation, LeakyReLU, Concatenate, Lambda

import tensorflow as tf


def UpConvBlock(x, upscale_factor=2, filters=32, kernel=3, stride=1, bias=True, \
                padding="SAME", activation='relu', mode='nearest', name='Upsample_c1'):
  
  if isinstance(activation, str):
    activation = Activation(activation)

  if mode == 'nearest':
    mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
  elif mode == 'bilinear':
    mode = tf.image.ResizeMethod.BILINEAR

  upsample = Lambda(lambda x: tf.image.resize_images(x, K.shape(x)[1:3]*2, method=mode))(x)
  x = Conv2D(filters, kernel, strides=(stride,stride), use_bias=bias, padding=padding, name=name)(upsample)
  x = activation(x)
  return x

def ResidualDenseConv5Block(x, filters=32, kernel=3, stride=1, bias=True, padding='SAME', activation=LeakyReLU(alpha=0.3), alpha=0.2, name="RDCB"):
  if isinstance(activation, str):
    activation = Activation(activation)
    
  x1 = Conv2D(filters, kernel, strides=(stride,stride),use_bias=bias, padding=padding, name=name+'_c1')(x)
  x1 = activation(x1)
  _x = Concatenate(axis=-1)([x,x1])
  
  x2 = Conv2D(filters, kernel, strides=(stride,stride),use_bias=bias, padding=padding, name=name+'_c2')(_x)
  x2 = activation(x2)
  _x = Concatenate(axis=-1)([x,x1,x2])
  
  x3 = Conv2D(filters, kernel, strides=(stride,stride),use_bias=bias, padding=padding, name=name+'_c3')(_x)
  x3 = activation(x3)
  _x = Concatenate(axis=-1)([x,x1,x2,x3])
  
  x4 = Conv2D(filters, kernel, strides=(stride,stride),use_bias=bias, padding=padding, name=name+'_c4')(_x)
  x4 = activation(x4)
  _x = Concatenate(axis=-1)([x,x1,x2,x3,x4])
  
  x5 = Conv2D(filters, kernel, strides=(stride,stride),use_bias=bias, padding=padding, name=name+'_c5')(_x)
  x5 = activation(x5)
  
  x5 = Lambda(lambda y: y[0]*alpha+y[1])([x5,x])
  
  return x5

def RRDBlock(x, filters=32, kernel=3, stride=1, bias=True, padding='SAME', activation=LeakyReLU(alpha=0.3), alpha=0.2, name='RRDB'):
  _x = ResidualDenseConv5Block(x, filters=filters, kernel=3, stride=stride, bias=bias, padding=padding, activation=activation, alpha=alpha, name=name+'_1')
  _x = ResidualDenseConv5Block(_x, filters=filters, kernel=3, stride=stride, bias=bias, padding=padding, activation=activation, alpha=alpha, name=name+'_2')
  _x = ResidualDenseConv5Block(_x, filters=filters, kernel=3, stride=stride, bias=bias, padding=padding, activation=activation, alpha=alpha, name=name+'_3')
  _x = Lambda(lambda y: y[0]*alpha+y[1])([_x,x])
  
  return _x
