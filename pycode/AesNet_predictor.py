# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import pandas as pd

# for include pycode
import sys
import os
sys.path.insert(0,'pycode')

# quiet caffe
os.environ['GLOG_minloglevel'] = '2'

import caffe
from caffe import layers as L
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import aestheticNet

caffe.set_device(0)  # if we have multiple GPUs, pick the first one

model_def = aestheticNet.caffenet_aes_test()
model_weights = "models/AesNet_CaffeNet.caffemodel"

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('models/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# these lines is for those images that have four dimensions
checked_image = caffe.io.load_image(sys.argv[1])
if (len(checked_image.shape)==4):
    checked_image = checked_image[0]

net.blobs['data'].data[0] = transformer.preprocess('data',checked_image)

net.forward()
print(net.blobs['probs'].data[0,1])