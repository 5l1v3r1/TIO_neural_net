from __future__ import absolute_import, division, print_function

import imageio
# TensorFlow and tf.keras
import tensorflow as tf
import tensorflow_datasets as tfds

from consts import img_dim, imgs_folder
# Helper libraries
from utils import label_number_to_name

print(tf.__version__)

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# tf_flowers total = 3,670
# img_dim = 50


# See available datasets
print(tfds.list_builders())

class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
ds_all = tfds.load(name="tf_flowers", split="train")
ds_all = ds_all.shuffle(buffer_size=100)

for i in range(3670):
    if i % 100 == 0:
        print("processing " + str(i))
    mnist_example, = ds_all.take(1)
    image, label = mnist_example["image"], mnist_example["label"]
    image = image / 255
    image = tf.image.resize_images(image, (img_dim, img_dim))
    # image = image.reshape(1, 50, 50)
    # image = image_fit.resize_to_fit(image, 12, 22)
    # image = tf.image.rgb_to_grayscale(image)
    label_name = label_number_to_name(label)
    imageio.imwrite(imgs_folder + "/" + label_name + "/" + str(i) + ".jpg", image)
