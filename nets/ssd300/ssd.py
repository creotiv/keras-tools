"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.models import Model

from ssd_layers import Normalize
from ssd_layers import PriorBox

from keras.applications.resnet50 import ResNet50


def SSD300(input_shape=(300, 300, 3), num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    net = {}
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

####################################################################################
    # zerro-padding need for backward compatibility
    x = ZeroPadding2D((3, 3))(input_tensor)
    model = ResNet50(include_top=False, input_tensor=x)
    for l in model.layers:
        l.trainable = False
    resnet_out = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pool5v')(model.get_layer('activation_49').output)
    net['conv3_4'] = model.get_layer("activation_22").output
    net['conv4_6'] = model.get_layer("activation_40").output

    # if not K.is_keras_tensor(input_tensor):
    #     net['input'] = Input(tensor=input_tensor)
    # else:
    #     net['input'] = input_tensor
    # if K.image_dim_ordering() == 'tf':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    # # Block 1
    # x = ZeroPadding2D((3, 3))(net['input'])
    # net['conv1'] = Conv2D(64, 7, 7, strides=(2, 2), name='conv1')(x)
    # net['bn_conv1'] = BatchNormalization(axis=bn_axis, name='bn_conv1')(net['conv1'])
    # x = Activation('relu')(net['bn_conv1'])
    # x = ZeroPadding2D((1, 1))(x)
    # net['pool1'] = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # # Block 2
    # net['conv2_1'] = conv_block(net['pool1'], 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # net['conv2_2'] = identity_block(net['conv2_1'], 3, [64, 64, 256], stage=2, block='b')
    # net['conv2_3'] = identity_block(net['conv2_2'], 3, [64, 64, 256], stage=2, block='c')

    # # Block 3
    # net['conv3_1'] = conv_block(net['conv2_3'], 3, [128, 128, 512], stage=3, block='a')
    # net['conv3_2'] = identity_block(net['conv3_1'], 3, [128, 128, 512], stage=3, block='b')
    # net['conv3_3'] = identity_block(net['conv3_2'], 3, [128, 128, 512], stage=3, block='c')
    # net['conv3_4'] = identity_block(net['conv3_3'], 3, [128, 128, 512], stage=3, block='d')

    # # Block 4
    # net['conv4_1'] = conv_block(net['conv3_4'], 3, [256, 256, 1024], stage=4, block='a')
    # net['conv4_2'] = identity_block(net['conv4_1'], 3, [256, 256, 1024], stage=4, block='b')
    # net['conv4_3'] = identity_block(net['conv4_2'], 3, [256, 256, 1024], stage=4, block='c')
    # net['conv4_4'] = identity_block(net['conv4_3'], 3, [256, 256, 1024], stage=4, block='d')
    # net['conv4_5'] = identity_block(net['conv4_4'], 3, [256, 256, 1024], stage=4, block='e')
    # net['conv4_6'] = identity_block(net['conv4_5'], 3, [256, 256, 1024], stage=4, block='f')

    # # Block 5
    # net['conv5_1'] = conv_block(net['conv4_6'], 3, [512, 512, 2048], stage=5, block='a')
    # net['conv5_2'] = identity_block(net['conv5_1'], 3, [512, 512, 2048], stage=5, block='b')
    # net['conv5_3'] = identity_block(net['conv5_2'], 3, [512, 512, 2048], stage=5, block='c')

    # # net['pool5'] = AveragePooling2D((7, 7), name='pool5')(net['conv5_3'])
    # # resnet uses this map directly onto the classification (top layer)
    # # we will use the VGG pooling instead, which provides an appropriately sized input to fc6
    # net['pool5v'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
    #                              name='pool5v')(net['conv5_3'])


# END ResNet50
#####################################################################################

    # FC6
    net['fc6'] = Conv2D(1024, (3, 3), dilation_rate=(6, 6),
                        activation='relu', padding='same',
                        name='fc6')(resnet_out)
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    net['fc7'] = Conv2D(1024, (1, 1), activation='relu',
                        padding='same', name='fc7')(net['fc6'])
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    net['conv6_1'] = Conv2D(256, (1, 1), activation='relu',
                            padding='same',
                            name='conv6_1')(net['fc7'])

    #net['conv6_2'] = ZeroPadding2D()(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, (3, 3), strides=(2, 2),
                            activation='relu', padding='same',
                            name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = Conv2D(128, (1, 1), activation='relu',
                            padding='same',
                            name='conv7_1')(net['conv6_2'])
    #net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                            activation='relu', padding='same',
                            name='conv7_2')(net['conv7_1'])
    # Block 8
    net['conv8_1'] = Conv2D(128, (1, 1), activation='relu',
                            padding='same',
                            name='conv8_1')(net['conv7_2'])
    #net['conv8_2'] = ZeroPadding2D()(net['conv8_1'])
    net['conv8_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                            activation='relu', padding='same',
                            name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])

    # Prediction from conv3_4 (still called conv4_3 in the remainder)
    # Will clean this up after training tests
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv3_4'])
    num_priors = 3
    x = Conv2D(num_priors * 4, (3, 3), padding='same',
               name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    net['conv4_3_norm_mbox_loc_flat'] = flatten(net['conv4_3_norm_mbox_loc'])
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same',
               name=name)(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])

    # Prediction from conv4_6 -- again, will replace after train test
    num_priors = 6
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, (3, 3),
                                 padding='same',
                                 name='fc7_mbox_loc')(net['conv4_6'])
    flatten = Flatten(name='fc7_mbox_loc_flat')
    net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, (3, 3),
                                  padding='same',
                                  name=name)(net['conv4_6'])  # changed from fc7
    flatten = Flatten(name='fc7_mbox_conf_flat')
    net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')

    # Another change from fc7
    net['fc7_mbox_priorbox'] = priorbox(net['conv4_6'])

    # Prediction from this fc7 (it will still be called 6_2)
    # project it so that its channels are 512, as bounding box data
    net['fc7_mbox_pre'] = Conv2D(512, (1, 1), activation='relu',
                                 padding='same', name='fc7_mbox_pre')(net['fc7'])
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same',
               name='conv6_2_mbox_loc')(net['fc7_mbox_pre'])
    net['conv6_2_mbox_loc'] = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same',
               name=name)(net['fc7_mbox_pre'])  # changed from conv6_2
    net['conv6_2_mbox_conf'] = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['fc7_mbox_pre'])  # changed from conv6_2

    # Prediction from conv6_2
    # Project it down to 256
    # (old conv7_2)
    net['conv6_2_mbox_pre'] = Conv2D(256, (1, 1), activation='relu',
                                     padding='same', name='conv6_2_mbox_pre')(net['conv6_2'])
    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same',
               name='conv7_2_mbox_loc')(net['conv6_2_mbox_pre'])
    net['conv7_2_mbox_loc'] = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same',
               name=name)(net['conv6_2_mbox_pre'])  # changed from conv7_2
    net['conv7_2_mbox_conf'] = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    # old conv7_2
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv6_2_mbox_pre'])
    # Prediction from conv7_2
    # old (conv8_2)
    # no projections needed

    num_priors = 6
    x = Conv2D(num_priors * 4, (3, 3), padding='same',
               name='conv8_2_mbox_loc')(net['conv7_2'])
    net['conv8_2_mbox_loc'] = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, (3, 3), padding='same',
               name=name)(net['conv7_2'])  # changed from conv8_2
    net['conv8_2_mbox_conf'] = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')

    net['conv8_2_mbox_priorbox'] = priorbox(net['conv7_2'])  # changed from conv8_2
    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_loc_flat'] = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(net['pool6'])
    net['pool6_mbox_conf_flat'] = x
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    net['pool6_reshaped'] = Reshape(target_shape,
                                    name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    # Gather all predictions
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([
        net['conv4_3_norm_mbox_loc_flat'],
        net['fc7_mbox_loc_flat'],
        net['conv6_2_mbox_loc_flat'],
        net['conv7_2_mbox_loc_flat'],
        net['conv8_2_mbox_loc_flat'],
        net['pool6_mbox_loc_flat']])
    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([
        net['conv4_3_norm_mbox_conf_flat'],
        net['fc7_mbox_conf_flat'],
        net['conv6_2_mbox_conf_flat'],
        net['conv7_2_mbox_conf_flat'],
        net['conv8_2_mbox_conf_flat'],
        net['pool6_mbox_conf_flat']])
    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([
        net['conv4_3_norm_mbox_priorbox'],
        net['fc7_mbox_priorbox'],
        net['conv6_2_mbox_priorbox'],
        net['conv7_2_mbox_priorbox'],
        net['conv8_2_mbox_priorbox'],
        net['pool6_mbox_priorbox']])
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    net['mbox_loc'] = Reshape((num_boxes, 4),
                              name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax',
                                  name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = Concatenate(axis=2, name='predictions')([
        net['mbox_loc'],
        net['mbox_conf'],
        net['mbox_priorbox']])
    model = Model(input_tensor, net['predictions'])
    return model
