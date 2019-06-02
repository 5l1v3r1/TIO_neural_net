import os

import imageio
import numpy as np
import tensorflow as tf

from consts import imgs_folder, img_dim, print_debug


def label_number_to_name(label):
    label_namee = "error"
    if int(label) == 0:
        label_namee = 'dandelion'
    elif int(label) == 1:
        label_namee = 'daisy'
    elif int(label) == 2:
        label_namee = 'tulips'
    elif int(label) == 3:
        label_namee = 'sunflowers'
    elif int(label) == 4:
        label_namee = 'roses'
    return label_namee


def label_name_to_number(label):
    label_namee = -1
    if label == 'dandelion':
        label_namee = 0
    elif label == 'daisy':
        label_namee = 1
    elif label == 'tulips':
        label_namee = 2
    elif label == 'sunflowers':
        label_namee = 3
    elif label == 'roses':
        label_namee = 4
    return label_namee


def load_flowers(start=0.0, end=1.0):
    dataset_all = []
    rootDir = imgs_folder
    for dirName, subdirList, fileList in os.walk(rootDir):
        if print_debug:
            print('Found directory: %s' % dirName)
        for dirname in subdirList:
            if print_debug:
                print('\t%s' % dirname)
            nested_load_flowers(dataset_all, dirName, dirname, start, end)
        # for fname in fileList:
        #     print('\t%s' % fname)
    # dataset_all = sorted(dataset_all, key=lambda x: x[1][0][0][0])
    return dataset_all


def nested_load_flowers(dataset_all, dirName, dirname, start, end):
    for dirName, subdirList, fileList in os.walk(dirName + "/" + dirname):
        from_i = int(len(fileList) * start)
        to_i = int(len(fileList) * end)
        print("taking {} from {} to {}".format(dirname, from_i, to_i))
        files = fileList[from_i:to_i]
        for fname in files:
            if print_debug:
                print('\t%s' % fname)
            dataset_all.append((dirname, imageio.imread(dirName + "/" + fname)))


def prepare_img_and_label(entry):
    label, image = entry
    image = prepare_image(image)
    label = label_name_to_number(label)
    return label, image


def prepare_image(image):
    image = np.asarray(image) / 255
    image = tf.image.resize_images(image, (img_dim, img_dim))
    image = image.numpy()
    return image
