from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from consts import model_name
from plot_utils import plot_image, plot_value_array
from utils import load_flowers, prepare_img_and_label

# TensorFlow and tf.keras

print(tf.__version__)

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()


# tf_flowers total = 3,670
# img_dim = 50


# See available datasets
print(tfds.list_builders())

class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
# Construct a tf.data.Dataset
# ds_train, ds_test = tfds.load(name="horses_or_humans", split=["train", "test"])
# ds_all = tfds.load(name="tf_flowers", split="train")
# ds_all = ds_all.shuffle(buffer_size=100)
# dataset_all = load_flowers()

train_images = []
train_labels = []
test_images = []
test_labels = []
ds_test = []
# split_point = int(len(dataset_all) * split_ratio)
# end_point = len(dataset_all)
# split_point = 2
# end_point = 4
dataset_train = load_flowers(0, 0.75)
dataset_test = load_flowers(0.75, 1)
for i in range(dataset_train.__len__()):
    if i % 100 == 0:
        print("processing " + str(i))
    label, image = prepare_img_and_label(dataset_train[i])
    train_images.append(image)
    train_labels.append(label)

for i in range(dataset_test.__len__()):
    if i % 100 == 0:
        print("processing " + str(i))
    label, image = prepare_img_and_label(dataset_test[i])
    test_images.append(image)
    test_labels.append(label)

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# mnist_example, = ds_all.take(1)
# image, label = mnist_example["image"], mnist_example["label"]

# plt.figure()
# plt.imshow(image)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
# plt.show()
# print("Label: %d" % label.numpy())

# model = keras.Sequential([
#     keras.layers.Conv2D(64, kernel_size=(20,20), activation='linear', input_shape=(img_dim, img_dim, 3), padding='same'),
#     keras.layers.LeakyReLU(alpha=0.1),
#     keras.layers.MaxPooling2D(pool_size=(5,5), padding='same'),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(128, kernel_size=(10,10), activation='linear', input_shape=(img_dim, img_dim, 3), padding='same'),
#     keras.layers.LeakyReLU(alpha=0.1),
#     keras.layers.MaxPooling2D(pool_size=(4,4), padding='same'),
#     keras.layers.Dropout(0.25),
#     keras.layers.Conv2D(512, kernel_size=(5, 5), activation='linear', input_shape=(img_dim, img_dim, 3), padding='same'),
#     keras.layers.LeakyReLU(alpha=0.1),
#     keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'),
#     keras.layers.Dropout(0.4),
#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.LeakyReLU(alpha=0.1),
#     keras.layers.Dropout(0.3),
#     keras.layers.Dense(len(class_names), activation=tf.nn.softmax)
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model = keras.models.load_model(model_name)

# tensor_train_images = tf.convert_to_tensor(train_images)
# tensor_train_images = tf.reshape(tensor_train_images, shape=(tensor_train_images.shape[0], 50, 50))
# tensor_train_labels = tf.convert_to_tensor(train_labels)
# # tensor_train_labels = tf.reshape(tensor_train_images, shape=(tensor_train_labels.shape[0], 5))
# dataset = tf.data.Dataset.from_tensor_slices((tensor_train_images, tensor_train_labels))
# images, labels = dataset.make_one_shot_iterator().get_next()
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

model.fit(train_images, train_labels, epochs=15)

model.save(model_name)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
np.argmax(predictions[0])

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images, class_names)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()
#
# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images, class_names)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
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

x = 0
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
