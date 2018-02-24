import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
from tqdm import tqdm
import pickle
import skimage.io


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class PortraitConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "portrait_seg"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (50, 93, 174, 313, 600)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 16

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    USE_MINI_MASK = False


class PortraitDataset(utils.Dataset):
    def load_portrait(self, dataset_dir, subset):
        # Add classes
        self.add_class("portrait", 1, "portrait")

        # Path
        image_dir = os.path.join(dataset_dir, subset)
        image_ids = []
        # Add images
        g = os.walk(image_dir)
        for path,_,filelist in g:
            for filename in filelist:
                image_ids.append(os.path.join(path, filename))

        for i in image_ids:
            self.add_image(
                "portrait", image_id=i.split('/')[-1],
                path=i)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        mask_path = '/home/yaopengfei/Desktop/portrait/images_mask/' + info['id'].split('.')[0] + '_mask.mat'

        assert mask_path is not None

        import scipy.io as sio
        mask = sio.loadmat(mask_path)['mask']

        # _, ax = plt.subplots(1)
        # ax.imshow(mask)
        # plt.show()
        
        a,b = mask.shape

        if a==0 or b==0:
            print('No mask found!!')
            input()

        mask = np.reshape(mask, (a,b,1))
        class_ids = np.array([1], dtype=np.int32)

        return mask, class_ids

if __name__ == '__main__':

    GPU_NO = '2'
    GPU_USE = .5

    #set gpu NO.
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NO
    import pickle
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config0 = tf.ConfigProto()
    config0.gpu_options.per_process_gpu_memory_fraction = GPU_USE
    set_session(tf.Session(config=config0))
    
    config = PortraitConfig()
    config.display()

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    #epoch
    n1 = 50
    n2 = 110

    #run mode
    training = True
    infer = False

    if training:
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = PortraitDataset()
        dataset_train.load_portrait('/home/yaopengfei/Desktop/portrait', "train")
        dataset_train.prepare()
        print("%d images added to train dataset."%len(dataset_train.image_info))

        # Validation dataset
        dataset_val = PortraitDataset()
        dataset_val.load_portrait('/home/yaopengfei/Desktop/portrait', "val")
        dataset_val.prepare()
        print("%d images added to val dataset."%len(dataset_val.image_info))


    if training:
        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
        model.keras_model.summary()

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last()[1], by_name=True)

        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        result1 = model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=n1, 
                    layers='heads')

        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also 
        # pass a regular expression to select which layers to
        # train by name pattern.
        result2 = model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=n2, 
                    layers="all")

        output = open('result1.pkl', 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(result1, output)
        output.close()
        output = open('result2.pkl', 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(result2, output)
        output.close()
