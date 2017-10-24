import keras
import keras.backend as K
import tensorflow as tf
import yaml
import os
import math
import numpy as np
from datadiff.tools import assert_equal


class TFCheckpointCallback(keras.callbacks.Callback):

    def __init__(self, saver, sess, path):
        self.saver = saver
        self.sess = sess
        self.path

    def on_epoch_end(self, epoch, logs=None):
        self.saver.save(self.sess, self.path, global_step=epoch)


class ConfigSaver(object):

    def __init__(self, path='./', variables={}):
        self._version = 0
        self._path = path
        self._dict = {k.upper(): v for k, v in variables.items()}

        self._get_current_version()
        self._save_if_differ()

    def version(self):
        return self._version

    def add_comment(self, txt):
        self._dict['__COMMENT__'] = txt
        self._save()

    def _save_if_differ(self):
        tmp = self._load()
        if tmp:
            try:
                assert_equal(self._dict, tmp)
            except Exception as e:
                self._version += 1
                self._save()
        else:
            self._save()

    def _get_current_version(self):
        self._version = 1
        t = sorted([int(f.split('.')[0].split('_')[1]) for f in os.listdir(self._path) if f.startswith('config_')])
        if t:
            last = t[-1]
            self._version = last

    def _save(self):
        with open(os.path.join(self._path, "config_%s.yaml" % self._version), 'w') as fp:
            yaml.dump(self._dict, fp, default_flow_style=False)

    def _load(self, version=None):
        if not version:
            version = self._version
        try:
            with open(os.path.join(self._path, "config_%s.yaml" % version)) as fp:
                d = yaml.load(fp)
                if '__COMMENT__' in d:
                    del d['__COMMENT__']
                return d
        except:
            return {}

    def __setattr__(self, key, value):
        if key in ['_dict', '_version', '_path']:
            return super(ConfigSaver, self).__setattr__(key, value)
        raise Exception('Configuration immutable')

    def __getattr__(self, key):
        if key in ['_dict', '_version', '_path']:
            return getattr(self, key)
        else:
            return getattr(self, '_dict').get(key.upper())


def imgs_side_by_side(imgs):
    num = imgs.shape[0]
    ch = imgs.shape[-1]
    h = imgs.shape[1]
    w = imgs.shape[2]

    edge = int(math.ceil(math.sqrt(num)))
    new_w = edge * w
    new_h = edge * h
    res = np.zeros((new_h, new_w, ch))
    res.fill(255)

    i = 0
    for y in range(edge):
        for x in range(edge):
            if i >= num:
                break
            res[y * h:y * h + h, x * w:x * w + w] = imgs[i]
            i += 1

    return res
