# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for store the results
import pickle
import gzip

# for include pycode
import sys
import os
sys.path.insert(0,'pycode')

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
caffe_root = '/opt/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
from caffe import layers as L
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import aestheticNet
from preprocess import utilities 

import utilsData
from sklearn.metrics import roc_auc_score, accuracy_score

data = pickle.load(gzip.open('packages/AVA_info.pklz','rb',2))
data.loc[:,'id'] = data['id'].apply(str)
classes = np.array(data.sort_values(['id']).loc[:,'Class'])
ids = np.array(data.sort_values(['id']).loc[:,'id'])
aux_list =  np.array(['/home/frubio/AVA/AVADataset/{:}.jpg'.format(i) for i in ids])

np.random.seed(1000)
random_idx = np.random.randint(3, size=len(ids))

train_files = aux_list[random_idx <= 1]
train_classes = classes[random_idx <= 1]
# Class balanced
new_indexes = utilities.balance_class(train_classes)
train_classes = train_classes[new_indexes]
train_files = train_files[new_indexes]

test_files = aux_list[random_idx == 2]
test_classes = classes[random_idx == 2]

# This lines are for test the models with the same test as the networks
data_for_train_models = pd.DataFrame(data.sort_values(['id']).loc[random_idx <= 1],copy=True)
data_for_train_models.reset_index(inplace=True)
data_for_train_models = data_for_train_models.loc[new_indexes]
train_indexes_for_models = np.array(data_for_train_models['index'])

data_for_test_models = pd.DataFrame(data.sort_values(['id']).loc[random_idx == 2],copy=True)
data_for_test_models.reset_index(inplace=True)
test_indexes_for_models = data_for_test_models['index']

model_def = aestheticNet.caffenet_aes_test()
model_weights = "models/AesNet_CaffeNet_iter_30000.caffemodel"

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

num_images = test_files.shape[0]
output_prob = np.zeros(num_images)
batch_size = net.blobs['data'].data.shape[0]

num_forwards = np.ceil(num_images/batch_size).astype(int)

for i in range(0,num_forwards):
    images_step = i*batch_size
    
    if (i == num_forwards - 1):
        images_to_process = num_images - images_step
    else:
        images_to_process = batch_size
            
    for j in range(0,images_to_process):
        # these lines is for those images that have four dimensions
        checked_image = caffe.io.load_image(test_files[j+images_step])
        if (len(checked_image.shape)==4):
            checked_image = checked_image[0]
                
        net.blobs['data'].data[j] = transformer.preprocess('data',checked_image)
        
    net.forward()
    output_prob[images_step:images_step+images_to_process] = net.blobs['probs'].data[0:images_to_process,1]
    
results = {}
results['balanced'] = utilsData.balanced_accuracy(test_classes, output_prob)
results['AUC'] = roc_auc_score(test_classes, output_prob)
results['accuracy'] = accuracy_score(test_classes, (output_prob >= 0.5).astype(int))

pickle.dump(results, gzip.open( "results/AesNet_CaffeNet_iter_30000.pklz", "wb" ), 2)