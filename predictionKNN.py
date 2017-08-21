#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:56:01 2017

@author: sourish
"""

# import library to use pickled model
from sklearn.externals import joblib

# Test Image stored as array
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Import library to load model
from keras.models import load_model

# Load previously saved classifier model
classifier = load_model('Dog_Cat.h5')

# Show Architecture of the model -- note the layer's names
classifier.summary()

# Input will be same as original Model
inputs = classifier.input 

# Output will be collected after flattening features, i.e. feature extraction
outputs = classifier.get_layer('flatten_1').output

# Create new model from intermediate layer
from keras.models import Model

intermediate_layer_model = Model (inputs,outputs)

# Show summary of new model

intermediate_layer_model.summary()

# Extract Features for test image
test_features = intermediate_layer_model.predict(test_image)


# Get original training data as array
trainData = joblib.load('traningArray.pkl')
#scale = joblib.load('dogs_cats_scaling.pkl')
# Scaling test images
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Fit the scalar on training data
train_features = sc_X.fit_transform(intermediate_layer_model.predict(trainData))
# transform the test features
test_features = sc_X.transform(test_features)


# Dimensionality reduction using Principal Component Analysis
from sklearn.decomposition import PCA

# After analysis we have seen 445 components (features) actually consist of 65% of variance of the data
# We are using 445 components as we had used in training phase
pca = PCA(n_components=445)
train_features = pca.fit_transform (train_features)
explain = pca.explained_variance_ratio_
test_features = pca.transform (test_features)

# Load Neighbor Model
nbrs = joblib.load('neighbourModel.pkl')

# Predict 3 Nearest Neigbors on test features
distances,indices = nbrs.kneighbors(test_features)

#show results
image.array_to_img(test_image[0])
image.array_to_img(trainData[indices[0][0]])
image.array_to_img(trainData[indices[0][1]])
image.array_to_img(trainData[indices[0][2]])
