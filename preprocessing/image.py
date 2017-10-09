from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from six.moves import range
import os
import multiprocessing.pool
from functools import partial
from scipy.misc import imresize, imsave

import keras.backend as K
from keras.preprocessing.image import (Iterator,
                                       _count_valid_files_in_directory,
                                       img_to_array,
                                       load_img,
                                       array_to_img,
                                       _list_valid_filenames_in_directory)
from keras.preprocessing.image import ImageDataGenerator as ImageDataGeneratorBase

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None


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
            binary_mask=False)


if __name__ == "__main__":
    import cv2
    # data_gen_args = dict(rotation_range=15.,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2)
    # ig = ImageDataGenerator(**data_gen_args)
    # d = ig.flow_from_directory('./mask/', batch_size=1,
    #                            class_mode='categorical', seed=1, color_mode="mask")
    # npy = d.next()
    # d = ig.flow_from_directory('./img/', batch_size=1, class_mode='categorical', seed=1)
    # img = d.next()
    # print (npy[1])
    # img_alpha = np.concatenate([img, npy], axis=-1)
    # imsave('img_gen_test.png', img_alpha)

    # data_gen_args = dict(rotation_range=15.,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2)
    # ig = ImageDataGenerator(**data_gen_args)
    # d = ig.flow_from_directory('/mnt/course/datasets/portraits/masks/', batch_size=1,
    #                            class_mode=None, seed=1, color_mode="mask")
    # npy = d.next()[0]
    # d = ig.flow_from_directory('/mnt/course/datasets/portraits/imgs/', batch_size=1, class_mode=None, seed=1)
    # img = d.next()[0]
    # img_alpha = np.concatenate([img, npy], axis=-1)
    # imsave('img_gen_test.png', img_alpha)
