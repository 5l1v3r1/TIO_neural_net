from __future__ import absolute_import, division, print_function

import imageio

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import plot_image, plot_value_array, plot_one_predicted

import easygui
path = easygui.fileopenbox()


# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

model = keras.models.load_model('path_to_my_model.h5')

# See available datasets
print(tfds.list_builders())

class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
# Construct a tf.data.Dataset
# ds_train, ds_test = tfds.load(name="horses_or_humans", split=["train", "test"])
# ds_all = tfds.load(name="tf_flowers", split="train")

# Build your input pipeline
# ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
# for features in ds_train.take(1):
#   image, label = features["image"], features["label"]

# tf_flowers total = 3,670

train_images = []
train_labels = []
test_images = []
test_labels = [0]
ds_test = []
split_point = int(3670 * 0.75)
split_point = 300
end_point = 20
# for i in range(split_point):
#     mnist_example, = ds_all.take(1)
#     image, label = mnist_example["image"], mnist_example["label"]
#     image = image / 255
#     image = tf.image.resize_images(image, (50, 50))
#     # image = image.reshape(1, 50, 50)
#     # image = image_fit.resize_to_fit(image, 12, 22)
#     image = tf.image.rgb_to_grayscale(image)
#     image = image.numpy()
#     train_images.append(image)
#     train_labels.append(label)

# path = r"C:\Users\wojciech.liebert\tensorflow_datasets\downloads\extracted\TAR_GZ.downl.tenso.org_examp_image_flowe_photoTFSs55Ear_4To2XDT2UOcd1b8b4KWLRk5acYPj5ZXZw.tgz\flower_photos\daisy"
# path = path + r"\21652746_cc379e0eea_m.jpg"
image = imageio.imread(path)
image = np.asarray(image)
image = image / 255
image = tf.image.resize_images(image, (50, 50))
image = tf.image.rgb_to_grayscale(image)
image = image.numpy()

test_images.append(image)
test_images = np.array(test_images)

predictions = model.predict(test_images)
np.argmax(predictions[0])


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_one_predicted(i, predictions, test_labels, test_images, class_names)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
#     plot_image(i, predictions, test_labels, test_images, class_names)
#     plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
#     plot_value_array(i, predictions, test_labels)
# plt.show()
