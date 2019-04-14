import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn import cluster
from sklearn.cluster import KMeans
import gdal as gdal
import glob
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from skimage import color
from skimage import io
%matplotlib inline

## importing dataset -- greyscale
import os
hairy_path = "C:/Users/green/Downloads/CS3244/hairy_root"
hairy_root_grey = {}
for filename in os.listdir(hairy_path):
    hairy_root_grey[filename] = io.imread(hairy_path+ "/" + filename, as_gray = True)

hairy_root_ravel = [value.ravel() for value in hairy_root_grey.values()]

## set seed for reproducibility
np.random.seed(3244)

## declaring the model -- different number of clusters
kmeans2 = KMeans(n_clusters = 2)
kmeans3 = KMeans(n_clusters = 3)
kmeans4 = KMeans(n_clusters = 4)

#############
## kmeans2 ##
#############
kmeans2.fit(hairy_root_ravel)
kmeans2_centroids = kmeans2.cluster_centers_
kmeans2_labels = kmeans2.labels_

##contains a dictionary of the image name as the key and the label (ie which cluster) as the value.
hairy_root_labelled2 = {}
i = 0
for key in hairy_root_grey.keys():
    hairy_root_labelled2[key] = kmeans2_labels[i]
    i+=1

## separate each cluster, to contain filenames that belong to each cluster
kmeans2_group0 = []
kmeans2_group1 = []

for key in hairy_root_labelled2.keys():
    if hairy_root_labelled2[key] == 0:
        kmeans2_group0.append(key)
    else:
        kmeans2_group1.append(key)

## printing the information out
print(len(kmeans2_group0)) ## 299 ## number of files clustered into group0.
print(len(kmeans2_group1)) ## 450

print(kmeans2_group0)
print(kmeans2_group1)

#############
## kmeans3 ##
#############
kmeans3.fit(hairy_root_ravel)
kmeans3_centroids = kmeans3.cluster_centers_
kmeans3_labels = kmeans3.labels_

##contains a dictionary of the image name as the key and the label (ie which cluster) as the value.
hairy_root_labelled3 = {}
i = 0
for key in hairy_root_grey.keys():
    hairy_root_labelled3[key] = kmeans3_labels[i]
    i+=1

## separate each cluster, to contain filenames that belong in each cluster
kmeans3_group0 = []
kmeans3_group1 = []
kmeans3_group2 = []
for key in hairy_root_labelled3.keys():
    if hairy_root_labelled3[key] == 0:
        kmeans3_group0.append(key)
    elif hairy_root_labelled3[key] == 1:
        kmeans3_group1.append(key)
    else:
        kmeans3_group2.append(key)

## printing the information out
print(len(kmeans3_group0)) ##230
print(len(kmeans3_group1)) ##290
print(len(kmeans3_group2)) ##229

print(kmeans3_group0)
print(kmeans3_group1)
print(kmeans3_group2)

#############
## kmeans4 ##
#############
kmeans4.fit(hairy_root_ravel)
kmeans4_centroids = kmeans4.cluster_centers_
kmeans4_labels = kmeans4.labels_

##contains a dictionary of the image name as the key and the label (ie which cluster) as the value.
hairy_root_labelled4 = {}
i = 0
for key in hairy_root_grey.keys():
    hairy_root_labelled4[key] = kmeans4_labels[i]
    i+=1

## separate each cluster to contain filenames in respective cluster
kmeans4_group0 = []
kmeans4_group1 = []
kmeans4_group2 = []
kmeans4_group3 = []
for key in hairy_root_labelled4.keys():
    if hairy_root_labelled4[key] == 0:
        kmeans4_group0.append(key)
    elif hairy_root_labelled4[key] == 1:
        kmeans4_group1.append(key)
    elif hairy_root_labelled4[key] == 2:
        kmeans4_group2.append(key)
    else:
        kmeans4_group3.append(key)

## printing out the information
print(len(kmeans4_group0)) ##221
print(len(kmeans4_group1)) ##220
print(len(kmeans4_group2)) ##152
print(len(kmeans4_group3)) ##156

print(kmeans4_group0)
print(kmeans4_group1)
print(kmeans4_group2)
print(kmeans4_group3)
