"""Keras 2.0 implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import AveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.applications.resnet50 import ResNet50

from ssd_layers import Normalize
from ssd_layers import PriorBox

from math import ceil


def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized


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
    prev_layer = BatchNormalization(momentum=0.95, epsilon=1e-5, name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Lambda(Interp, arguments={'shape': feature_map_shape})(prev_layer)
    return prev_layer


def build_psp(res, input_shape):
    feature_map_size = (60, 60)  # tuple(int(ceil(input_dim / 5.0)) for input_dim in input_shape)
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


def SSD(input_shape=(300, 300, 3), num_classes=21, segmentation_head=False, depth_head=False):
    """SSD architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3).
        num_classes: Number of classes including background.

    conv3_4  conv4_6   fc7   conv6_2   conv7_2   pool6
      +       +         +      +           +       +
      |       |         |      |           |       |
      |       |         v      v           |       |
      |       |                            |       |
      |       |    +----------------+      |       |
      |       +--> |                | <----+       |
      |            |  Concatenate   |              |
      +----------> |                |  <-----------+
                   +-------+--------+
                           |
                           v
                       prediction


    # References
        SSD: https://arxiv.org/abs/1512.02325
        Rainbow SSD: https://arxiv.org/abs/1705.09587
    """
    net = {}
    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

####################################################################################
    # zerro-padding need for backward compatibility
    x = ZeroPadding2D((3, 3))(input_tensor)
    model = ResNet50(include_top=False, input_tensor=x)
    # resnet_out = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='pool5v')(model.get_layer('activation_49').output)
    resnet_out = MaxPooling2D((3, 3), strides=(1, 1), padding='same',
                              name='pool5v')(model.get_layer('activation_49').output)

    net['conv3_4'] = model.get_layer("activation_22").output
    net['conv4_6'] = model.get_layer("activation_40").output

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

    net['conv6_2'] = Conv2D(512, (3, 3), strides=(2, 2),
                            activation='relu', padding='same',
                            name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = Conv2D(128, (1, 1), activation='relu',
                            padding='same',
                            name='conv7_1')(net['conv6_2'])

    net['conv7_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                            activation='relu', padding='same',
                            name='conv7_2')(net['conv7_1'])
    # Block 8
    net['conv8_1'] = Conv2D(128, (1, 1), activation='relu',
                            padding='same',
                            name='conv8_1')(net['conv7_2'])

    net['conv8_2'] = Conv2D(256, (3, 3), strides=(2, 2),
                            activation='relu', padding='same',
                            name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])

    ###########################################################################
    # Segmentation PSP ########################################################
    if segmentation_head:
        x = Lambda(Interp, arguments={'shape': (60, 60)})(net['conv3_4'])
        psp = build_psp(x, input_shape[:2])

        # segmentation
        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="seg_conv1_3", use_bias=False)(psp)
        x = BatchNormalization(momentum=0.95, epsilon=1e-5, name="seg_conv1_3_bn")(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        x = Conv2D(num_classes, (1, 1), strides=(1, 1), name="seg_conv_last")(x)
        x = Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1])})(x)
        segmentation = Activation('sigmoid', name='segmentation')(x)

    if depth_head:
        # depth map
        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="depth_conv1_3", use_bias=False)(psp)
        x = BatchNormalization(momentum=0.95, epsilon=1e-5, name="depth_conv1_3_bn")(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="depth_conv2_3", use_bias=False)(x)
        x = BatchNormalization(momentum=0.95, epsilon=1e-5, name="depth_conv2_3_bn")(x)
        x = Activation('relu')(x)
        x = Conv2D(1, (3, 3), strides=(1, 1), padding="same", name="depth_conv2_3", use_bias=False)(x)
        x = Lambda(Interp, arguments={'shape': (input_shape[0], input_shape[1])})(x)
        depth_map = Activation('relu', name="depth_map")(x)

    ###########################################################################

    asp0 = [1. / 2, 1, 1., 2.]
    asp1 = [1. / 3, 1. / 2, 1, 1., 2., 3.]
    scales = [0.1, 0.2, 0.38, 0.56, 0.74, 0.92, 1.1]

    ###########################################################################
    # CLASSIFIER:1 LAYER: conv3_4 #############################################

    num_priors = len(asp0)

    cl1_input = Normalize(20, name='conv3_4_norm')(net['conv3_4'])

    x = Conv2D(num_priors * 4, (3, 3), strides=(1, 1), dilation_rate=(2, 2),
               padding='same', name='conv3_4_norm_mbox_loc')(cl1_input)

    x = Flatten(name='conv3_4_norm_mbox_loc_flat')(x)
    net['conv3_4_norm_mbox_loc_flat'] = x

    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name="conv3_4_norm_mbox_conf")(cl1_input)

    x = Flatten(name='conv3_4_norm_mbox_conf_flat')(x)
    net['conv3_4_norm_mbox_conf_flat'] = x

    x = PriorBox(img_size, scales[0] * img_size[0], aspect_ratios=asp0,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 name='conv3_4_norm_mbox_priorbox')(cl1_input)
    net['conv3_4_norm_mbox_priorbox'] = x

    ###########################################################################
    # CLASSIFIER:2 LAYER: conv4_6 #############################################

    num_priors = len(asp1)
    cl2_input = net['conv4_6']

    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='fc7_mbox_loc')(cl2_input)

    x = Flatten(name='fc7_mbox_loc_flat')(x)
    net['fc7_mbox_loc_flat'] = x

    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name="fc7_mbox_conf")(cl2_input)

    x = Flatten(name='fc7_mbox_conf_flat')(x)
    net['fc7_mbox_conf_flat'] = x

    x = PriorBox(img_size, scales[1] * img_size[0], max_size=scales[2] * img_size[0], aspect_ratios=asp1,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 name='fc7_mbox_priorbox')(cl2_input)

    net['fc7_mbox_priorbox'] = x

    ###########################################################################
    # CLASSIFIER:3 LAYER: fc7 #################################################

    num_priors = len(asp1)

    cl3_input = Conv2D(512, (1, 1), activation='relu', padding='same', name='fc7_mbox_pre')(net['fc7'])

    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(cl3_input)

    x = Flatten(name='conv6_2_mbox_loc_flat')(x)
    net['conv6_2_mbox_loc_flat'] = x

    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name="conv6_2_mbox_conf")(cl3_input)

    x = Flatten(name='conv6_2_mbox_conf_flat')(x)
    net['conv6_2_mbox_conf_flat'] = x

    x = PriorBox(img_size, scales[2] * img_size[0], max_size=scales[3] * img_size[0], aspect_ratios=asp1,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 name='conv6_2_mbox_priorbox')(cl3_input)

    net['conv6_2_mbox_priorbox'] = x

    ###########################################################################
    # CLASSIFIER:4 LAYER: conv6_2 #############################################

    num_priors = len(asp1)

    cl4_input = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_2_mbox_pre')(net['conv6_2'])

    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv7_2_mbox_loc')(cl4_input)

    x = Flatten(name='conv7_2_mbox_loc_flat')(x)
    net['conv7_2_mbox_loc_flat'] = x

    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name="conv7_2_mbox_conf")(cl4_input)

    x = Flatten(name='conv7_2_mbox_conf_flat')(x)
    net['conv7_2_mbox_conf_flat'] = x

    x = PriorBox(img_size, scales[3] * img_size[0], max_size=scales[4] * img_size[0], aspect_ratios=asp1,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 name='conv7_2_mbox_priorbox')(cl4_input)

    net['conv7_2_mbox_priorbox'] = x

    ###########################################################################
    # CLASSIFIER:5 LAYER: conv7_2 #############################################

    num_priors = len(asp1)
    cl5_input = net['conv7_2']

    x = Conv2D(num_priors * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(cl5_input)

    x = Flatten(name='conv8_2_mbox_loc_flat')(x)
    net['conv8_2_mbox_loc_flat'] = x

    x = Conv2D(num_priors * num_classes, (3, 3), padding='same', name="conv8_2_mbox_conf")(cl5_input)

    x = Flatten(name='conv8_2_mbox_conf_flat')(x)
    net['conv8_2_mbox_conf_flat'] = x

    x = PriorBox(img_size, scales[4] * img_size[0], max_size=scales[5] * img_size[0], aspect_ratios=asp1,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 name='conv8_2_mbox_priorbox')(cl5_input)

    net['conv8_2_mbox_priorbox'] = x

    ###########################################################################
    # CLASSIFIER:6 LAYER: pool6 ###############################################

    num_priors = len(asp0)
    cl6_input = net['pool6']

    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(cl6_input)
    net['pool6_mbox_loc_flat'] = x

    x = Dense(num_priors * num_classes, name="pool6_mbox_conf_flat")(cl6_input)
    net['pool6_mbox_conf_flat'] = x

    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    x = Reshape(target_shape, name='pool6_reshaped')(cl6_input)

    x = PriorBox(img_size, scales[5] * img_size[0], max_size=scales[6] * img_size[0], aspect_ratios=asp0,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 name='pool6_mbox_priorbox')(x)

    net['pool6_mbox_priorbox'] = x

    ###########################################################################

    # Gather all predictions
    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([
        net['conv3_4_norm_mbox_loc_flat'],
        net['fc7_mbox_loc_flat'],
        net['conv6_2_mbox_loc_flat'],
        net['conv7_2_mbox_loc_flat'],
        net['conv8_2_mbox_loc_flat'],
        net['pool6_mbox_loc_flat']])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([
        net['conv3_4_norm_mbox_conf_flat'],
        net['fc7_mbox_conf_flat'],
        net['conv6_2_mbox_conf_flat'],
        net['conv7_2_mbox_conf_flat'],
        net['conv8_2_mbox_conf_flat'],
        net['pool6_mbox_conf_flat']])

    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox')([
        net['conv3_4_norm_mbox_priorbox'],
        net['fc7_mbox_priorbox'],
        net['conv6_2_mbox_priorbox'],
        net['conv7_2_mbox_priorbox'],
        net['conv8_2_mbox_priorbox'],
        net['pool6_mbox_priorbox']])

    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4

    net['mbox_loc'] = Reshape((num_boxes, 4), name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation('softmax', name='mbox_conf_final')(net['mbox_conf'])

    ssd_out = Concatenate(axis=2, name='ssd_out')([
        net['mbox_loc'],
        net['mbox_conf'],
        net['mbox_priorbox']])

    if not segmentation_head and not depth_head:
        model = Model(input_tensor, ssd_out)
    else:
        out = [ssd_out]
        if segmentation_head:
            out.append(segmentation)
        if depth_head:
            out.append(depth_map)
        model = Model(input_tensor, out)
    return model
