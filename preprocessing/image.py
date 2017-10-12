from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from six.moves import range
import os
import cv2
import multiprocessing.pool
from functools import partial
from scipy.misc import imresize, imsave
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
from keras.preprocessing.image import (Iterator,
                                       _count_valid_files_in_directory,
                                       img_to_array,
                                       load_img,
                                       array_to_img,
                                       _list_valid_filenames_in_directory,
                                       transform_matrix_offset_center,
                                       apply_transform,
                                       random_channel_shift,
                                       flip_axis)
from keras.preprocessing.image import ImageDataGenerator as ImageDataGeneratorBase

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


def center_crop(x, center_crop_size, data_format, **kwargs):
    if data_format == 'channels_first':
        centerh, centerw = x.shape[1] // 2, x.shape[2] // 2
    elif data_format == 'channels_last':
        centerh, centerw = x.shape[0] // 2, x.shape[1] // 2
    lh, lw = center_crop_size[0] // 2, center_crop_size[1] // 2
    rh, rw = center_crop_size[0] - lh, center_crop_size[1] - lw

    h_start, h_end = centerh - lh, centerh + rh
    w_start, w_end = centerw - lw, centerw + rw
    if data_format == 'channels_first':
        return x[:, h_start:h_end, w_start:w_end]
    elif data_format == 'channels_last':
        return x[h_start:h_end, w_start:w_end, :]


def pair_center_crop(x, y, center_crop_size, data_format, **kwargs):
    if data_format == 'channels_first':
        centerh, centerw = x.shape[1] // 2, x.shape[2] // 2
    elif data_format == 'channels_last':
        centerh, centerw = x.shape[0] // 2, x.shape[1] // 2
    lh, lw = center_crop_size[0] // 2, center_crop_size[1] // 2
    rh, rw = center_crop_size[0] - lh, center_crop_size[1] - lw

    h_start, h_end = centerh - lh, centerh + rh
    w_start, w_end = centerw - lw, centerw + rw
    if data_format == 'channels_first':
        return x[:, h_start:h_end, w_start:w_end], \
            y[:, h_start:h_end, w_start:w_end]
    elif data_format == 'channels_last':
        return x[h_start:h_end, w_start:w_end, :], \
            y[h_start:h_end, w_start:w_end, :]


def random_crop(x, random_crop_size, data_format, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    if data_format == 'channels_first':
        h, w = x.shape[1], x.shape[2]
    elif data_format == 'channels_last':
        h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)

    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    if data_format == 'channels_first':
        return x[:, h_start:h_end, w_start:w_end]
    elif data_format == 'channels_last':
        return x[h_start:h_end, w_start:w_end, :]


def pair_random_crop(x, y, random_crop_size, data_format, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    if data_format == 'channels_first':
        h, w = x.shape[1], x.shape[2]
    elif data_format == 'channels_last':
        h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)

    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    if data_format == 'channels_first':
        return x[:, h_start:h_end, w_start:w_end], y[:, h_start:h_end, h_start:h_end]
    elif data_format == 'channels_last':
        return x[h_start:h_end, w_start:w_end, :], y[h_start:h_end, w_start:w_end, :]


class SegDirectoryIterator(Iterator):
    '''
    Users need to ensure that all files exist.
    Label images should be png images where pixel values represents class number.
    find images -name *.jpg > images.txt
    find labels -name *.png > labels.txt
    for a file name 2011_002920.jpg, each row should contain 2011_002920
    file_path: location of train.txt, or val.txt in PASCAL VOC2012 format,
        listing image file path components without extension
    data_dir: location of image files referred to by file in file_path
    label_dir: location of label files
    data_suffix: image file extension, such as `.jpg` or `.png`
    label_suffix: label file suffix, such as `.png`, or `.npy`
    loss_shape: shape to use when applying loss function to the label data
    '''

    def __init__(self, file_path, seg_data_generator,
                 data_dir, data_suffix,
                 label_dir, label_suffix, classes, ignore_label=255,
                 crop_mode='none', label_cval=255, pad_size=None,
                 target_size=None, color_mode='rgb',
                 data_format='default', class_mode='sparse',
                 batch_size=1, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 loss_shape=None):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.file_path = file_path
        self.data_dir = data_dir
        self.data_suffix = data_suffix
        self.label_suffix = label_suffix
        self.label_dir = label_dir
        self.classes = classes
        self.seg_data_generator = seg_data_generator
        self.target_size = tuple(target_size)
        self.ignore_label = ignore_label
        self.crop_mode = crop_mode
        self.label_cval = label_cval
        self.pad_size = pad_size
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        self.nb_label_ch = 1
        self.loss_shape = loss_shape

        if (self.label_suffix == '.npy') or (self.label_suffix == 'npy'):
            self.label_file_format = 'npy'
        else:
            self.label_file_format = 'img'
        if target_size:
            if self.color_mode == 'rgb':
                if self.data_format == 'channels_last':
                    self.image_shape = self.target_size + (3,)
                else:
                    self.image_shape = (3,) + self.target_size
            else:
                if self.data_format == 'channels_last':
                    self.image_shape = self.target_size + (1,)
                else:
                    self.image_shape = (1,) + self.target_size
            if self.data_format == 'channels_last':
                self.label_shape = self.target_size + (self.nb_label_ch,)
            else:
                self.label_shape = (self.nb_label_ch,) + self.target_size
        elif batch_size != 1:
            raise ValueError(
                'Batch size must be 1 when target image size is undetermined')
        else:
            self.image_shape = None
            self.label_shape = None
        if class_mode not in {'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of '
                             '"sparse", or None.')
        self.class_mode = class_mode
        if save_to_dir:
            self.palette = None
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'npy'}

        # build lists for data files and label files
        self.data_files = []
        self.label_files = []
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        self.nb_sample = len(lines)
        for line in lines:
            line = line.strip('\n')
            self.data_files.append(line + data_suffix)
            self.label_files.append(line + label_suffix)
        super(SegDirectoryIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # The transformation of images is not under thread lock so it can be
        # done in parallel
        if self.target_size:
            # TODO(ahundt) make dtype properly configurable
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            if self.loss_shape is None and self.label_file_format is 'img':
                batch_y = np.zeros((current_batch_size,) + self.label_shape,
                                   dtype=int)
            elif self.loss_shape is None:
                batch_y = np.zeros((current_batch_size,) + self.label_shape)
            else:
                batch_y = np.zeros((current_batch_size,) + self.loss_shape,
                                   dtype=np.uint8)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data and labels
        for i, j in enumerate(index_array):
            data_file = self.data_files[j]
            label_file = self.label_files[j]
            img_file_format = 'img'
            img = load_img(os.path.join(self.data_dir, data_file),
                           grayscale=grayscale, target_size=None)
            label_filepath = os.path.join(self.label_dir, label_file)

            if self.label_file_format == 'npy':
                y = np.load(label_filepath)
            else:
                label = pil_image.open(label_filepath)
                if self.save_to_dir and self.palette is None:
                    self.palette = label.palette

            # do padding
            if self.target_size:
                if self.crop_mode != 'none':
                    x = img_to_array(img, data_format=self.data_format)
                    if self.label_file_format is not 'npy':
                        y = img_to_array(
                            label, data_format=self.data_format).astype(int)
                    img_w, img_h = img.size
                    if self.pad_size:
                        pad_w = max(self.pad_size[1] - img_w, 0)
                        pad_h = max(self.pad_size[0] - img_h, 0)
                    else:
                        pad_w = max(self.target_size[1] - img_w, 0)
                        pad_h = max(self.target_size[0] - img_h, 0)
                    if self.data_format == 'channels_first':
                        x = np.lib.pad(x, ((0, 0), (pad_h / 2, pad_h - pad_h / 2),
                                           (pad_w / 2, pad_w - pad_w / 2)), 'constant', constant_values=0.)
                        y = np.lib.pad(y, ((0, 0), (pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2)),
                                       'constant', constant_values=self.label_cval)
                    elif self.data_format == 'channels_last':
                        x = np.lib.pad(x, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w -
                                                                            pad_w / 2), (0, 0)), 'constant', constant_values=0.)
                        y = np.lib.pad(y, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2),
                                           (0, 0)), 'constant', constant_values=self.label_cval)
                else:
                    x = img_to_array(img.resize((self.target_size[1], self.target_size[0]),
                                                pil_image.BILINEAR),
                                     data_format=self.data_format)
                    if self.label_file_format is not 'npy':
                        y = img_to_array(label.resize((self.target_size[1], self.target_size[
                                         0]), pil_image.NEAREST), data_format=self.data_format).astype(int)
                    else:
                        print('ERROR: resize not implemented for label npy file')

            if self.target_size is None:
                batch_x = np.zeros((current_batch_size,) + x.shape)
                if self.loss_shape is not None:
                    batch_y = np.zeros((current_batch_size,) + self.loss_shape)
                else:
                    batch_y = np.zeros((current_batch_size,) + y.shape)

            x, y = self.seg_data_generator.random_transform(x, y)
            x = self.seg_data_generator.standardize(x)

            if self.ignore_label:
                y[np.where(y == self.ignore_label)] = self.classes

            if self.loss_shape is not None:
                y = np.reshape(y, self.loss_shape)

            batch_x[i] = x
            batch_y[i] = y
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                label = batch_y[i][:, :, 0].astype('uint8')
                label[np.where(label == self.classes)] = self.ignore_label
                label = pil_image.fromarray(label, mode='P')
                label.palette = self.palette
                fname = '{prefix}_{index}_{hash}'.format(prefix=self.save_prefix,
                                                         index=current_index + i,
                                                         hash=np.random.randint(1e4))
                img.save(os.path.join(self.save_to_dir, 'img_' +
                                      fname + '.{format}'.format(format=self.save_format)))
                label.save(os.path.join(self.save_to_dir,
                                        'label_' + fname + '.png'))
        # return
        batch_x = preprocess_input(batch_x)
        if self.class_mode == 'sparse':
            return batch_x, batch_y
        else:
            return batch_x


class SegDataGenerator(object):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 channelwise_center=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 label_cval=255,
                 crop_mode='none',
                 crop_size=(0, 0),
                 pad_size=None,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 data_format='default'):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.__dict__.update(locals())
        self.mean = None
        self.ch_mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if data_format not in {'channels_last', 'channels_first'}:
            raise Exception('data_format should be channels_last (channel after row and '
                            'column) or channels_first (channel before row and column). '
                            'Received arg: ', data_format)
        if crop_mode not in {'none', 'random', 'center'}:
            raise Exception('crop_mode should be "none" or "random" or "center" '
                            'Received arg: ', crop_mode)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if data_format == 'channels_last':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow_from_directory(self, file_path, data_dir, data_suffix,
                            label_dir, label_suffix, classes,
                            ignore_label=255,
                            target_size=None, color_mode='rgb',
                            class_mode='sparse',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg',
                            loss_shape=None):
        if self.crop_mode == 'random' or self.crop_mode == 'center':
            target_size = self.crop_size
        return SegDirectoryIterator(
            file_path, self,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes, ignore_label=ignore_label,
            crop_mode=self.crop_mode, label_cval=self.label_cval,
            pad_size=self.pad_size,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format, class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format,
            loss_shape=loss_shape)

    def standardize(self, x):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.channelwise_center:
            x -= self.ch_mean
        return x

    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        if self.crop_mode == 'none':
            crop_size = (x.shape[img_row_index], x.shape[img_col_index])
        else:
            crop_size = self.crop_size

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            # * x.shape[img_row_index]
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * crop_size[0]
        else:
            tx = 0

        if self.width_shift_range:
            # * x.shape[img_col_index]
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * crop_size[1]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        if self.zoom_maintain_shape:
            zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)

        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode='constant', cval=self.label_cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(
                x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        if self.crop_mode == 'center':
            x, y = pair_center_crop(x, y, self.crop_size, self.data_format)
        elif self.crop_mode == 'random':
            x, y = pair_random_crop(x, y, self.crop_size, self.data_format)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center and featurewise_std_normalization
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

    def set_ch_mean(self, ch_mean):
        self.ch_mean = ch_mean


class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`, `"mask"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set)
        binary_mask: Wheter to save values in [0,1] or in [0,255] interval for mask. 
            For second using lanczos interpolation.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 binary_mask=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.binary_mask = binary_mask
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale', 'mask'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        elif self.color_mode == 'grayscale' or self.color_mode == 'mask':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'npy'}

        # first, count the number of samples and classes
        self.samples = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)
        if not classes:
            classes = [""]
        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        results = []

        self.filenames = []
        if not classes[0]:
            results.append(pool.apply_async(self._list_valid_filenames_in_directory, (directory, white_list_formats)))

            for res in results:
                filenames = res.get()
                self.filenames += filenames
        else:
            self.classes = np.zeros((self.samples,), dtype='int32')
            i = 0
            for dirpath in (os.path.join(directory, subdir) for subdir in classes):
                results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                                (dirpath, white_list_formats,
                                                 self.class_indices, follow_links)))
            for res in results:
                classes, filenames = res.get()
                self.classes[i:i + len(classes)] = classes
                self.filenames += filenames
                i += len(classes)
        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    @classmethod
    def _list_valid_filenames_in_directory(cls, dirpath, white_list_formats):
        res = []
        for f in sorted(os.listdir(dirpath)):
            if f.split('.')[-1].lower() in white_list_formats:
                res.append(f)
        return res

    def _load_file(self, path):
        grayscale = self.color_mode == 'grayscale'
        if self.color_mode == "mask":
            mask = np.load(path)
            if self.binary_mask:
                mask = imresize(mask, size=self.target_size, interp="nearest")
                mask[mask == 255] = 1
            else:
                mask *= 255
                mask = imresize(mask, size=self.target_size, interp="lanczos")
            if self.data_format == "channels_first":
                mask = np.expand_dims(mask, axis=0)
            else:
                mask = np.expand_dims(mask, axis=-1)
            return mask
        else:
            return img_to_array(load_img(path,
                                         grayscale=grayscale,
                                         target_size=self.target_size), data_format=self.data_format)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            x = self._load_file(os.path.join(self.directory, fname))
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class ImageDataGenerator(ImageDataGeneratorBase):

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            binary_mask=False):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            binary_mask=binary_mask)


def zip_gen(args, merge=[], merge_axis=-1, preprocess_func=None, preprocess_args=[0]):
    while True:
        tmp = []
        for i, p in enumerate(args):
            el = p.next()
            if preprocess_func and i in preprocess_args:
                el = preprocess_func(el)
            tmp.append(el)
        if merge:
            to_merge = []
            for m in merge:
                to_merge.append(tmp[m])
            out = []
            for m in range(len(tmp)):
                if m not in merge:
                    out.append(tmp[m])
            out.append(np.concatenate(to_merge, axis=merge_axis))
            yield tuple(out)
        else:
            yield tuple(tmp)
            
def image2masks(img, ignore_colors=[],color_to_lable={}):
    '''
        This is not very fast algorithm
    
        # Example?:
            
            labels = {
                (38.,  156.,  255.): "car",
                (255.,  59.,  54.): "person",
            }
            res = image2masks(img, ignore_colors=[(254.0, 249.0, 246.0)], color_to_lable=labels)
            print res['car'].shape
    '''
    size = img.shape[0]*img.shape[1]
    a = np.concatenate(img[:,:]).tolist()
    colors = np.array(list(set(map(lambda x: tuple(x), a)) - set(ignore_colors)))
    out = {}
    for i in range(len(colors)):
        indices = np.where(np.all(img == colors[i], axis=-1))
        m = np.zeros(img.shape[:2]+(1,),dtype=np.uint8)
        label = color_to_lable.get(tuple(colors[i]))
        m[indices] = 255
        out[label] = m
    return out

if __name__ == "__main__":
    pass
    # Example

    # ig1 = ImageDataGenerator()
    # ig2 = ImageDataGenerator()
    # ig3 = ImageDataGenerator()
    # ig4 = ImageDataGenerator()

    # imggen = ig1.flow_from_directory('/mnt/course/datasets/portraits/imgs/', binary_mask=True, target_size=(305,305), batch_size=1, class_mode=None, seed=1)
    # maskgen = ig2.flow_from_directory('/mnt/course/datasets/portraits/masks/', binary_mask=True, target_size=(305,305), batch_size=1, class_mode=None, color_mode="mask", seed=1)

    # traingen = zip_gen([imggen, maskgen])

    # imggen2 = ig3.flow_from_directory('/mnt/course/datasets/portraits/imgs/', binary_mask=True, target_size=(305,305), batch_size=1, class_mode=None, seed=2)
    # maskgen2 = ig4.flow_from_directory('/mnt/course/datasets/portraits/masks/', binary_mask=True, target_size=(305,305), batch_size=1, class_mode=None, color_mode="mask", seed=2)

    # testgen = zip_gen([imggen2, maskgen2])

    # model.fit_generator(
    #     generator=traingen, validation_data=testgen,
    #     steps_per_epoch=500,
    #     validation_steps=50,
    #     epochs=100,
    #     verbose=1
    # )
