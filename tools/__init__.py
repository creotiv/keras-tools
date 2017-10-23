import keras
import keras.backend as K
import tensorflow as tf
import json
import os
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
        with open(os.path.join(self._path, "config_%s.json" % self._version), 'w') as fp:
            fp.write(json.dumps(self._dict))

    def _load(self, version=None):
        if not version:
            version = self._version
        try:
            with open(os.path.join(self._path, "config_%s.json" % version)) as fp:
                d = json.loads(fp.read().strip())
                del d['__COMMENT__']
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

