""""
@author: Nitish
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

classifier = Sequential() # creating object of Sequential() function
#Adding convolution layer to capture all patterns of image and get featured map(also decreases size)
#32, 3,3 is filter size(aka feature detector) of 3x3. 32 feature maps will be created
#64x64 is pixel size of image for further processing, 3 is for 3 layers of same matrix(since it's a color image(rgb)))
#input_shape = (64, 64, 3); since it is a Tensorflow backend; input_shape = (3,64, 64) for  Theano
classifier.add(Convolution2D(32, 3,3, input_shape = (64, 64, 3),  activation= 'relu'))

#Adding Pooling layer to decrease size of matrix into half without loosing features of image.
#Unique values are taken into consideration at each step
#Taking max values of matrix at every iteration
classifier.add(MaxPooling2D(pool_size= (2,2)))

#Flatten the matrix, converting to 1-D array of a vector
#This vector will be input for further fully connected layers(ANN layers)

##Adding one more convonutional layer to increase accuracy on test data
classifier.add(Convolution2D(32, 3,3,  activation= 'relu'))
 #input_shape is not mentioned in above layer, since keras understands that it's taking input from previous max-pooled layer
classifier.add(MaxPooling2D(pool_size= (2,2)))


classifier.add(Flatten())
#Adding hidden and output layer
classifier.add(Dense(output_dim= 128, activation= 'relu'))
#output_dim= 128 is a hyper-parameter here. Should be not to small to loose info n not too large to increase computer complexity
#output_dim= 128; value should in power of 2
classifier.add(Dense(output_dim=1, activation= 'sigmoid'))
# compiling all layers now
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#Image Augmentation to avoid overfitting
#It is done to train model better with different combination, orientation and batch sizes of input images
#Usually done when we don't have enough images to train model for better prediction on test_data

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) # rescaling pixel sizes between 0 & 1. Pixels have sizes ranging from 0 to 255

train_data= train_datagen.flow_from_directory(
        'dataset2/training_set',
        target_size=(64, 64), #changed size as per previous  algorithm's defined sizes
        batch_size=32, #no. of images per batch for training
        class_mode='binary')

test_data = test_datagen.flow_from_directory(
        'dataset2/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_data,
        steps_per_epoch=8000,
        epochs=10,
        validation_data= test_data,
        validation_steps=2000)