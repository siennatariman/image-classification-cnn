# SRMT - CPE215_D01

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


# Obtaining and preparing the data
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# visualize images, use 4x4 grid

# for i in range(16):
#      plt.subplot(4,4,i+1)
#      plt.xticks([])
#      plt.yticks([])
#      plt.imshow(training_images[i], cmap=plt.cm.binary)
#      plt.xlabel(class_names[training_labels[i][0]])
#
# plt.show()

# building and training model
# to save resources
training_images = training_images[:60000]
training_labels = training_labels[:60000]
testing_images = testing_images[:120000]
testing_labels = testing_labels[:120000]

model = models.load_model('image_classifier.model')

img = cv.imread('bird1.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')

plt.show()