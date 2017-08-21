#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Referred from Deep Learning A-Z™: Hands-On Artificial Neural Networksin Udemy Course

# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution -- input shape 64x64 has been used to mitigate computational overhead
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN -- Feel free to try other optimizer parameter e.g. sgd etc.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# Image augmentation for mitigate overfitting with the noise and low data volume

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# For computational benefit the below model have used image target Size 64x64 and batch size = 32
# Class Mode is set to binary as there are only 2 classes, i.e. Binary Classification
# Create Training and Test data set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
# Train the CNN classifier with Training and Validation Set
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

# Show Model Architecture -- Note the layer's name as those will be used subsequently
classifier.summary()

# Save the model for future use
classifier.save('Dog_Cat.h5')

#Optional modular use -- delete the classifier
#del classifier

from keras.models import load_model

classifier = load_model('Dog_Cat.h5')

classifier.summary()

outputs = classifier.get_layer('flatten_1').output
inputs = classifier.input 

from keras.models import Model

intermediate_layer_model = Model (inputs,outputs)

intermediate_layer_model.summary()

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


from os import listdir

filenames_dog = listdir('dataset/training_set/dogs')

filenames_cat = listdir('dataset/training_set/cats')

from keras.preprocessing import image # for image preprocessing


train_images = [] #declare training image array

for i in range (len(filenames_dog)) :
    
    temp_image = image.load_img('dataset/training_set/dogs/'+filenames_dog[i],target_size=(64,64))
    
    temp_image = image.img_to_array(temp_image)
    
    train_images.append(temp_image)

for i in range (len(filenames_cat)) :
    
    temp_image = image.load_img('dataset/training_set/cats/'+filenames_cat[i],target_size=(64,64))
    
    temp_image = image.img_to_array(temp_image)
    
    train_images.append(temp_image)
    

    
import numpy as np #for data manupulation
    
train_images=np.array(train_images) 

filenames = listdir('dataset/training_set/cats')

from keras.preprocessing import image # for image preprocessing

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

result = sc_X.fit_transform(intermediate_layer_model.predict(train_images))

#test_result = sc_X.fit_transform(intermediate_layer_model.predict(test_image))

test_result = sc_X.transform(intermediate_layer_model.predict(test_image))

from sklearn.decomposition import PCA

pca = PCA(n_components=445)

result = pca.fit_transform (result)

explain = pca.explained_variance_ratio_

test_result = pca.transform(test_result)


from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(result)

distances,indices = nbrs.kneighbors(test_result)

from sklearn.externals import joblib

filename = 'neighbourModel.pkl'

joblib.dump(nbrs,filename)

del nbrs

nbrs = joblib.load('neighbourModel.pkl')

train_images[7597]

image.array_to_img(test_image[0])
image.array_to_img(train_images[indices[0][0]])
image.array_to_img(train_images[indices[0][1]])
image.array_to_img(train_images[indices[0][2]])
image.array_to_img(train_images[indices[0][3]])
image.array_to_img(train_images[indices[0][4]])