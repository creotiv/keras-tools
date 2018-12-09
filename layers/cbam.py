from keras.layers import (Dense, Multiply, GlobalMaxPooling2D, Conv2D, 
                          UpSampling2D, GlobalAveragePooling2D, Concatenate, 
                          Reshape, MaxPooling2D, AveragePooling2D, Activation, 
                          Add)
from keras import backend as K
from keras.engine.topology import Layer

class CBAM(Layer):
    """
        CBAM: Convolutional Block Attention Module
        https://arxiv.org/abs/1807.06521
    """
    def call(self, inp):
        H, W, C = tuple([int(x) for x in inp.get_shape()[1:]])
        # Channel attention
        x = GlobalMaxPooling2D()(inp)
        y = GlobalAveragePooling2D()(inp)
        o = Concatenate(axis=-1)([x,y])
        o = Dense(C*2)(o)
        o = Dense(C)(o)
        o = Dense(C*2)(o)
        o = Reshape((1,1,C, 2))(o)
        o = K.tile(o,[1,H,W,1,1])
        o = Add()([o[:,:,:,:,0],o[:,:,:,:,1]])
        o = Activation('sigmoid')(o)
        inp2 = Multiply()([o, inp])

        # Spatial attention
        x = MaxPooling2D()(inp2)
        y = AveragePooling2D()(inp2)
        o = Concatenate(axis=-1)([x,y])
        o = Conv2D(C, 3, padding='same', activation='relu')(o)
        o = UpSampling2D()(o)
        o = Conv2D(C, 3, padding='same', activation='sigmoid')(o)
        o = Multiply()([o, inp2])
        return inp2

    def compute_output_shape(self, input_shape):
        return input_shape
