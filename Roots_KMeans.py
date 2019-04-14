import numpy as np
import matplotlib.pyplot as plt
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
import random
get_ipython().run_line_magic('matplotlib', 'inline')

####################
## Import Dataset ##
####################
import os
roots_path = "C:/Users/green/Downloads/CS3244/allData"
all_root = {}
for filename in os.listdir(roots_path):
        all_root[filename] = io.imread(roots_path+ "/" + filename, as_gray = True)

all_root_ravel = {}
for key,value in all_root.items():
    all_root_ravel[key] = value.ravel()
    
root_ravel = [value.ravel() for value in all_root.values()]

##############
## Set Seed ##
##############
np.random.seed(3244) 


######################
## Train-Test Split ##
######################
all_root_names = list(all_root_ravel.keys()) ##obtain all the keys first
random.shuffle(all_root_names) ## random shuffle the keys
train_data = all_root_names[:1000] ## select the first 1000 keys as training data
train_ravel = [all_root_ravel[key] for key in train_data] ## obtain the ravelled form of the images in training set
test_data = all_root_names[1000:]## remaining keys make up the test data 
test_ravel = [all_root_ravel[key] for key in test_data] ## obtain the ravelled form of the images in test set

######################
## KMeans Modelling ##
######################
kmeans = KMeans(n_clusters = 2) ##declaring the model
kmeans.fit(train_ravel) ##fitting the model
kmeans_centroids = kmeans.cluster_centers_ ##obtain the centroids of the 2 clusters
kmeans_labels = kmeans.labels_ ##get the labels of each datapoint

## get the respective cluster labels of image name from the training set
train_root_label = {}
i = 0
for key in train_data:
    train_root_label[key] = kmeans_labels[i]
    i += 1

## count number of classifications for each hairy and nonhairy image item
count_a1= 0
count_a0 = 0
count_b1 = 0
count_b0 = 0
for key, value in train_root_label.items():
    if key[-5:] == "a.jpg":
        if value == 1:
            count_a1 += 1
        else:
            count_a0 += 1
    else:
        if value == 1:
            count_b1 += 1
        else:
            count_b0 += 1
print(count_a1) ## 202 ie the number of hairy roots misclassified as nonhairy
print(count_a0) ## 313 ie the number of hairy roots classified correctly
print(count_b1) ## 248 ie the number of nonhairy roots classified correctly
print(count_b0) ## 237 ie the number of nonhairy roots misclassified as hairy

print((count_a0 + count_b1)/1000)
## for training:
## let 0 be hairy and 1 be nonhairy
## classification rate = 0.561
## misclassification rate = 0.439

#############
## TESTING ##
#############
test_predicted_label = kmeans.predict(test_ravel)
test_root_label = {}
i = 0
for key in test_data:
    test_root_label[key] = test_predicted_label[i]
    i += 1

print(test_root_label)


# In[17]:


count_a1= 0
count_a0 = 0
count_b1 = 0
count_b0 = 0
for key, value in test_root_label.items():
    if key[-5:] == "a.jpg":
        if value == 1:
            count_a1 += 1
        else:
            count_a0 += 1
    else:
        if value == 1:
            count_b1 += 1
        else:
            count_b0 += 1
print(count_a1) ## 94 ie the number of hairy roots misclassified
print(count_a0) ## 139 ie the number of hairy roots classified correctly
print(count_b1) ## 126 ie the number of nonhairy roots classified correctly
print(count_b0) ## 137 ie the number of nonhairy roots misclassified

print((count_a0 + count_b1)/len(test_ravel))
## for test:
## classification rate = 0.53427
## misclassification rate = 0.46573

