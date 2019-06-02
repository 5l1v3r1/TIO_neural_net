from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import plot_image, plot_value_array

print(tf.__version__)

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
# Construct a tf.data.Dataset
# ds_train, ds_test = tfds.load(name="horses_or_humans", split=["train", "test"])
ds_all = tfds.load(name="tf_flowers", split="train")

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
split_point = int(3670 * 0.75)
split_point = 300
end_point = 20
for i in range(split_point):
    mnist_example, = ds_all.take(1)
    image, label = mnist_example["image"], mnist_example["label"]
    image = image / 255
    image = tf.image.resize_images(image, (50, 50))
    # image = image.reshape(1, 50, 50)
    # image = image_fit.resize_to_fit(image, 12, 22)
    image = tf.image.rgb_to_grayscale(image)
    image = image.numpy()
    train_images.append(image)
    train_labels.append(label)

for i in range(split_point, split_point + end_point):
    mnist_example, = ds_all.take(1)
    image, label = mnist_example["image"], mnist_example["label"]
    image = image / 255
    image = tf.image.resize_images(image, (50, 50))
    image = tf.image.rgb_to_grayscale(image)
    image = image.numpy()
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

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(50, 50, 1)),
    # keras.layers.InputLayer(input_shape=(50, 50, 1)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

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

model.fit(train_images, train_labels, epochs=5)

model.save('path_to_my_model.h5')

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
