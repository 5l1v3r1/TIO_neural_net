from __future__ import absolute_import, division, print_function

import random

import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from consts import model_name
from plot_utils import plot_image, plot_value_array
# tfds works in both Eager and Graph modes
from utils import prepare_img_and_label, load_flowers

tf.enable_eager_execution()


model = keras.models.load_model(model_name)

# See available datasets
# print(tfds.list_builders())

class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
# # Construct a tf.data.Dataset
# # ds_train, ds_test = tfds.load(name="horses_or_humans", split=["train", "test"])
# ds_all = tfds.load(name="tf_flowers", split="train")

# Build your input pipeline
# ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
# for features in ds_train.take(1):
#   image, label = features["image"], features["label"]

# tf_flowers total = 3,670

train_images = []
train_labels = []
test_images = []
test_labels = []
ds_test = []
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

# dataset_all = load_flowers()
# ds_all = ds_all.shuffle(buffer_size=100)
# for i in range(0, 20):
#     mnist_example, = ds_all.take(1)
#     image, label = mnist_example["image"], mnist_example["label"]
#     image = image / 255
#     image = tf.image.resize_images(image, (img_dim, img_dim))
#     image = tf.image.rgb_to_grayscale(image)
#     image = image.numpy()
#     test_images.append(image)
#     test_labels.append(label)

dataset_test = load_flowers(0.75, 1)
# dataset_test = load_flowers(0, 1)
for _ in range(25):
    i = random.randint(0, dataset_test.__len__() - 1)
# for i in range(dataset_test.__len__()):
    #     if i % 100 == 0:
    #         print("processing " + str(i))
    label, image = prepare_img_and_label(dataset_test[i])
    test_images.append(image)
    test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)


predictions = model.predict(test_images)
successes = np.sum([x[0] == x[1] for x in zip([np.argmax(x) for x in predictions], test_labels)])
success_ratio = successes / len(test_images)
print("success ratio: {}".format(success_ratio))

x = 0

# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images, class_names)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images, class_names)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()
