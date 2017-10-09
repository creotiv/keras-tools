from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D,AveragePooling2D,Conv2D, BatchNormalization, Input, UpSampling2D, Maximum, Lambda
from keras.models import Model
from math import ceil
from keras.layers.merge import Concatenate, Add
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized


def fast_module(x, id, filters=128, name="fast_module"):

    xl = Convolution2D(filters, (3, 1), padding='same', name=name + "_left_" + id)(x)
    xc = Convolution2D(filters, (1, 1), padding='valid', name=name + "_center_" + id)(x)
    xr = Convolution2D(filters, (3, 1), padding='same', name=name + "_right_" + id)(x)

    x = Concatenate(axis=-1, name=name + "_concatenate_" + id)([xl, xc, xr])
    x = BatchNormalization()(x)
    x = Activation('relu', name=name + "_relu_" + id)(x)
    x = MaxPooling2D()(x)

    return x


def reverse_fast_module(x, id, filters=128, name="fast_module"):

    x = UpSampling2D()(x)

    xl = Convolution2D(filters, (3, 1), padding='same', name=name + "_left_" + id)(x)
    xc = Convolution2D(filters, (1, 1), padding='valid', name=name + "_center_" + id)(x)
    xr = Convolution2D(filters, (3, 1), padding='same', name=name + "_right_" + id)(x)

    x = Concatenate(axis=-1, name=name + "_concatenate_" + id)([xl, xc, xr])
    x = BatchNormalization()(x)
    x = Activation('relu', name=name + "_relu_" + id)(x)

    return x


def interp_block(prev_layer, level, feature_map_shape, str_lvl=1):
    str_lvl = str(str_lvl)

    names = [
        "conv5_3_pool" + str_lvl + "_conv",
        "conv5_3_pool" + str_lvl + "_conv_bn"
    ]

    kernel = (10 * level, 10 * level)
    strides = (10 * level, 10 * level)
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Lambda(Interp, arguments={'shape': feature_map_shape})(prev_layer)
    return prev_layer


def build_psp(res, input_shape):
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)
    interp_block1 = interp_block(res, 6, feature_map_size, str_lvl=1)
    interp_block2 = interp_block(res, 3, feature_map_size, str_lvl=2)
    interp_block3 = interp_block(res, 2, feature_map_size, str_lvl=3)
    interp_block6 = interp_block(res, 1, feature_map_size, str_lvl=6)

    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def newnet(classes=1):
    input = Input((305, 305, 3), name="new_input")
    x1 = x = fast_module(input, '1', 32)
    x2 = x = fast_module(x, '2', 64)
    x3 = x = fast_module(x, '3', 128)
    x = fast_module(x, '4', 256)
    x = Add()([reverse_fast_module(x, '5', 128), x3])
    x = Add()([reverse_fast_module(x, '6', 64), x2])
    #x = Add()([reverse_fast_module(x, '7', 32), x1])
    x = Lambda(Interp, arguments={'shape': (60, 60)})(x)
    x = build_psp(x, (473, 473))

    x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv5_4", use_bias=False)(x)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Convolution2D(classes, (3, 3), padding='same', name="last_conv")(x)
    x = Lambda(Interp, arguments={'shape': (305, 305)})(x)
    x = Activation('sigmoid', name="sigmoid")(x)

    model = Model(input, x, name='newnet')

    return model
