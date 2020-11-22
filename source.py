from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
import argparse
import os
import cv2

# imagePaths = list(paths.list_images("E:/Face Mask Detector/dataset/"))
# data = []
# labels = []
# loop over the image paths
# f#or imagePath in imagePaths:
#	# extract the class label from the filename
#	label = imagePath.split(os.path.sep)[-2]
#	# load the input image (224x224) and preprocess it
#	image = load_img(imagePath, target_size=(224, 224))
#	image = img_to_array(image)
#	image = preprocess_input(image)

# update the data and labels lists, respectively#
#	data.append(image)
#	labels.append(label)
# convert the data and labels to NumPy arrays
# data = np.array(data, dtype="float32")
# labels = np.array(labels)
# print(labels)

# le = LabelEncoder()
# labels = le.fit_transform(labels)
# labels = to_categorical(labels)
# print(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Part 2 - Building the CNN

# Initialising the CNN
# cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Step 2 - Pooling
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
# cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
# cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
# cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
# cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
# cnn.fit(trainX, trainY, epochs = 20)

# cnn.save('E:/Face Mask Detector/mask_detector.h5', save_format="h5")
from tensorflow.keras.preprocessing import image

model = load_model('E:/Face Mask Detector/mask_detector.h5')
test_image = image.load_img('E:/Face Mask Detector/examples/example_5.png', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
if result[0] == 1:
    print('No Mask')
else:
    print('Mask')
