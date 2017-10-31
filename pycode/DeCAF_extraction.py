
# coding: utf-8

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt

# for store the results
from six.moves import cPickle as pickle
import gzip

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
import os
caffe_root = '/opt/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

caffe.set_device(0)

model_def = caffe_root + 'models/ResNet-152/ResNet-152-deploy.prototxt'
model_weights = caffe_root + 'models/ResNet-152/ResNet-152-model.caffemodel'

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

def extract_decaf_features(images_root,list_images,layer,net,transformer):
    # set the size of the input (we can skip this if we're happy
    # with the default; we can also change it later, e.g., for different batch sizes)
    batch, C, H, W = net.blobs['data'].shape
    
    num_images = len(list_images)
    
    output_shape = net.blobs[layer].shape
    output_shape[0] = num_images
    output = np.zeros(output_shape)
    
    count = 0
    while count < num_images:
        pre_count = count    
        for i in range(0,batch):
            if count >= num_images:
                i -= 1
                break
            try:
                image = caffe.io.load_image(images_root + list_images[count])
                if len(image.shape)>3:
                    image = image[0]
                transformed_image = transformer.preprocess('data', image)
                net.blobs['data'].data[i] = transformed_image
            except:
                print (count)
                raise
            count += 1
        
        net.forward()
    
        output[pre_count:count]=net.blobs[layer].data[0:i+1]
    
    return output

images_root = '/home/frubio/AVA/AVADataset/'
list_images = os.listdir(images_root)
list_images.sort()

decaf_features = extract_decaf_features(images_root,list_images,'pool5',net,transformer)
try:
    with open('total_pool5_ResNet.pklz', 'wb') as f:
        pickle.dump(decaf_features, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', 'total_pool5_ResNet.pklz', ':', e)
    
'''
decaf_features = extract_decaf_features(images_root,list_images,'fc6',net,transformer)
try:
    with open('total_fc6.pklz', 'wb') as f:
        pickle.dump(decaf_features, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', 'total_fc6.pklz', ':', e)
    

decaf_features = extract_decaf_features(images_root,list_images,'fc7',net,transformer)
try:
    with open('total_fc7.pklz', 'wb') as f:
        pickle.dump(decaf_features, f, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', 'total_fc7.pklz', ':', e)
''' 

