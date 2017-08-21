#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 22:51:05 2017

@author: sourish
"""
# import library to use pickled model
from sklearn.externals import joblib

result = joblib.load('dogs_cats_PCA.pkl')


# import library for calculating Nearest Neighbors
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(result)


# Pickle Nearest Neighbor Model
filename = 'neighbourModel.pkl'

joblib.dump(nbrs,filename)
