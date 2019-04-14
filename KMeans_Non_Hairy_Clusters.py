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

## importing the dataset as greyscale
import os
nonhairy_path = "C:/Users/green/Downloads/CS3244/non_hairy_root"
nonhairy_root_grey = {}
for filename in os.listdir(nonhairy_path):
    nonhairy_root_grey[filename] = io.imread(nonhairy_path+ "/" + filename, as_gray = True)

## flatten the array of each image -- will use this for kmeans clustering
nonhairy_root_ravel = [value.ravel() for value in nonhairy_root_grey.values()]

## set seed (for reproducibility)
np.random.seed(3244) 

## declaring the models -- different number of clusters
kmeans2 = KMeans(n_clusters = 2)
kmeans3 = KMeans(n_clusters = 3)
kmeans4 = KMeans(n_clusters = 4)

#############
## kmeans2 ##
#############
kmeans2.fit(nonhairy_root_ravel)
kmeans2_centroids = kmeans2.cluster_centers_
kmeans2_labels = kmeans2.labels_

##contains a dictionary of the image name as the key and the label (ie which cluster) as the value.
nonhairy_root_labelled2 = {} 
i = 0
for key in nonhairy_root_grey.keys():
    nonhairy_root_labelled2[key] = kmeans2_labels[i]
    i+=1

## storing respective filenames to each group it was clustered into
kmeans2_group0 = []
kmeans2_group1 = []

for key in nonhairy_root_labelled2.keys():
    if nonhairy_root_labelled2[key] == 0:
        kmeans2_group0.append(key)
    else:
        kmeans2_group1.append(key)
  
print(len(kmeans2_group0)) ## 405 ## number of roots clustered in group 0
print(len(kmeans2_group1)) ## 343 ## number of roots clustered in group 1

##print(kmeans2_group0) ## files that belong in group 0
##print(kmeans2_group1) ## files that belong in group 1


#############
## kmeans3 ##
#############
kmeans3.fit(nonhairy_root_ravel)
kmeans3_centroids = kmeans3.cluster_centers_
kmeans3_labels = kmeans3.labels_

##contains a dictionary of the image name as the key and the label (ie which cluster) as the value.
nonhairy_root_labelled3 = {} 
i = 0
for key in nonhairy_root_grey.keys():
    nonhairy_root_labelled3[key] = kmeans3_labels[i]
    i+=1

## storing respective filenames to each group it was clustered into
kmeans3_group0 = []
kmeans3_group1 = []
kmeans3_group2 = []

for key in nonhairy_root_labelled3.keys():
    if nonhairy_root_labelled3[key] == 0:
        kmeans3_group0.append(key)
    elif nonhairy_root_labelled3[key] == 1:
        kmeans3_group1.append(key)
    else:
        kmeans3_group2.append(key)

## doing the printing of the information we can gather
print(len(kmeans3_group0)) ## 198
print(len(kmeans3_group1)) ## 290
print(len(kmeans3_group2)) ## 260

print(kmeans3_group0) ## files that belong to group 0
print(kmeans3_group1) ## files that belong to group 1
print(kmeans3_group2) ## files that belong to group 2


#############
## kmeans4 ##
#############
kmeans4.fit(nonhairy_root_ravel)
kmeans4_centroids = kmeans4.cluster_centers_
kmeans4_labels = kmeans4.labels_

##contains a dictionary of the image name as the key and the label (ie which cluster) as the value.
nonhairy_root_labelled4 = {} 
i = 0
for key in nonhairy_root_grey.keys():
    nonhairy_root_labelled4[key] = kmeans4_labels[i]
    i+=1
    
## storing respective filenames to each group it was clustered into
kmeans4_group0 = []
kmeans4_group1 = []
kmeans4_group2 = []
kmeans4_group3 = []

for key in nonhairy_root_labelled4.keys():
    if nonhairy_root_labelled4[key] == 0:
        kmeans4_group0.append(key)
    elif nonhairy_root_labelled4[key] == 1:
        kmeans4_group1.append(key)
    elif nonhairy_root_labelled4[key] == 2:
        kmeans4_group2.append(key)
    else:
        kmeans4_group3.append(key)
      
## doing the printing of the information
print(len(kmeans4_group0)) ## 147
print(len(kmeans4_group1)) ## 233
print(len(kmeans4_group2)) ## 197
print(len(kmeans4_group3)) ## 171
    
print(kmeans4_group0)
print(kmeans4_group1)
print(kmeans4_group2)
print(kmeans4_group3)

