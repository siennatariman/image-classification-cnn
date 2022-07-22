# build neural network
model = models.Sequential()
#input layer CNN
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
     # 32 neurons, 3x3 convolution matrix, act. func. is relu, input shape is 32x32 pixels
#maxpool layer
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
#flatten input, make it one-dimensional
model.add(layers.Flatten())
# add two dense layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) #scales all the results so that it will add up to 1
     # distribution of probabilities

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images,testing_labels))
     # epochs = how often will the model see the same data over and over again

loss, accuracy = model.evaluate(testing_images, testing_labels)
#metrics
print(f"Loss: {loss}") # numerical value that indicates how wrong our model is/how off it is from the ideal result
print(f"Accuracy: {accuracy}") # how much percent of the testing examples were classified correctly

# save, so that we don't need to train the model again
model.save('image_classifier.model')
# training = 20,000; test = 40,000 = accuracy of 65%
# training = 60,000; test = 120,000 = accuracy of 70%
# realtively decent for a neural network with 10 possible classifications