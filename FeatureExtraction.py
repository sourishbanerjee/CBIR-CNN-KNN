#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:13:18 2017

@author: sourish
"""
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


# Save the Model for test feature extraction

#intermediate_layer_model.save('featureExatractor.h5')


# List the files for image lists for future iteration

from os import listdir

filenames_dog = listdir('dataset/training_set/dogs')

filenames_cat = listdir('dataset/training_set/cats')

# Create Image database array for all cats and dogs present in training folder

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

#Save raw training array before scaling and PCA for future testing
from sklearn.externals import joblib

filename = 'traningArray.pkl'

joblib.dump(train_images,filename)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

# Extract the features and scale them in one step
result = sc_X.fit_transform(intermediate_layer_model.predict(train_images))

# Pickle the scaling model for future testing
from sklearn.externals import joblib

filename = 'dogs_cats_scaling.pkl'

joblib.dump(result,filename)


# Dimensionality reduction using Principal Component Analysis
from sklearn.decomposition import PCA

# After analysis we have seen 445 components (features) actually consist of 65% of variance of the data
pca = PCA(n_components=445)

result = pca.fit_transform (result)

explain = pca.explained_variance_ratio_

# Pickle the PCA model for future testing
# from sklearn.externals import joblib

filename = 'dogs_cats_PCA.pkl'

joblib.dump(result,filename)


