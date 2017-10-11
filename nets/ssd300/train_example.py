import os
import sys

import ssd
import ssd_training
import ssd_utils

from pycocotools.coco import COCO

from keras.optimizers import Adam, SGD
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard

import numpy as np
from scipy.misc import imread
from scipy.misc import imresize

CLASSES = 6
LEARNING_RATE = 0.001
COCODIR = "/mnt/course/datasets/coco/train2017/"
PRIORS_PATH = "./prior_boxes.npy"
CATEGORIES = ['person', 'car', 'motorcycle', 'bus', 'truck']

MODEL = ssd.SSD300

def generator(batch_size):
    img_list = []
    for cat_id in cat_ids:
        img_list += cocodata.getImgIds(catIds=[cat_id])
    print "Images: ", len(img_list)
    n = 0
    while True:
        X = np.zeros((batch_size, 300, 300, 3), dtype=np.float)
        Y = []
        for i in range(0, batch_size):
            n = n % len(img_list)
            img_data = cocodata.loadImgs(img_list[n])[0]
            img_path = os.path.join(COCODIR, img_data['file_name'])
            anns = cocodata.imgToAnns[img_list[n]]
            img = imread(img_path, mode="RGB").astype('float32')
            _x = imresize(img, (300, 300, 3)).astype('float32')
            bbox_list = []
            for ann in anns:
                if ann['category_id'] not in cat_ids:
                    continue
                bbox = np.array(ann['bbox'], dtype=np.float)
                bbox[0] *= 1. / img_data['width'] # x
                bbox[1] *= 1. / img_data['height']  # y
                bbox[2] = bbox[0] + bbox[2] * (1. / img_data['width'])  # width
                bbox[3] = bbox[1] + bbox[3] * (1. / img_data['height'])  # height
                classes = np.zeros(CLASSES - 1)
                classes[coco_2_ids[ann['category_id']]] = 1
                bbox_list.append(np.concatenate([bbox, classes]))
            Y.append(bboxer.assign_boxes(np.array(bbox_list)))
            X[i] = _x
            n += 1
        Y = np.array(Y)
        yield preprocess_input(X), Y


def schedule(epoch, decay=0.9):
    return LEARNING_RATE * decay**(epoch)


def main():
    # removing current priors files as model will create new one in append mode
    if os.path.exists(PRIORS_PATH):
        os.remove(PRIORS_PATH)

    model = MODEL((300, 300, 3), num_classes=CLASSES)
    mbloss = ssd_training.MultiboxLoss(num_classes=CLASSES, neg_pos_ratio=2.0)

    for l in model.layers[:12]:
        l.trainable = False

    model.compile(optimizer=SGD(LEARNING_RATE, momentum=0.9, nesterov=True), metrics=['accuracy'], loss=mbloss.compute_loss)
    # model.load_weights('/weights/ssd300-0.39.hdf5', by_name=True)
    print(model.summary())

    priors = np.load(PRIORS_PATH)
    bboxer = ssd_utils.BBoxUtility(num_classes=CLASSES, priors=priors)

    cocodata = COCO('/mnt/course/datasets/coco/annotations/instances_train2017.json')
    cats = cocodata.loadCats(cocodata.getCatIds())
    cat_names  = labels = CATEGORIES
    cat_ids = []
    for name in cat_names:
        cat_ids += cocodata.getCatIds(catNms=[name])
    coco_2_ids = {_id: i for i, _id in enumerate(cat_ids)}

    g = generator(16)
    v = generator(16)

    model.fit_generator(
        generator=g, validation_data=v,
        steps_per_epoch=150,
        validation_steps=50,
        epochs=3000,
        verbose=1,
        callbacks=[
            ModelCheckpoint('/mnt/nikish/weights/ssd300-{val_acc:.2f}.hdf5',
                            verbose=1, monitor='val_acc', save_best_only=True, period=1),
            LearningRateScheduler(schedule)
        ]
    )

if __name__ == '__main__':
    main()
