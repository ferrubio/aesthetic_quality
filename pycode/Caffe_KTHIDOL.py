
# coding: utf-8

# # Classification: Instant Recognition with Caffe
# 
# In this example we'll classify an image with the bundled CaffeNet model (which is based on the network architecture of Krizhevsky et al. for ImageNet).
# 
# We'll compare CPU and GPU modes and then dig into the model to inspect features and the output.

# ### 1. Setup
# 
# * First, set up Python, `numpy`, and `matplotlib`.

# In[ ]:

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np

# for store the results
import pickle
import gzip

# In[ ]:

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
import os
caffe_root = '/opt/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.


# ### 2. Load net and set up input preprocessing
# 
# * Set Caffe to GPU mode (Tesla) and load the net from disk.

# In[ ]:

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
#caffe.set_mode_cpu()


# In[ ]:

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this, but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
# 
#     Our default CaffeNet is configured to take images in BGR format. Values are expected to start in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them. In addition, the channel dimension is expected as the first (_outermost_) dimension.
#     
#     As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the _innermost_ dimension, we are arranging for the needed transformations here.

# In[ ]:

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values


# In[ ]:

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# ### 3. Extract the features
# 
# * Now we're ready to perform classification. First we create an array with the files of the images.

# In[ ]:

images_root = '/home/frubio/KTH-IDOL/Minnie/%s/'
illumination = ['Cloudy1','Cloudy2','Cloudy3','Cloudy4','Night1','Night2','Night3','Night4','Sunny1','Sunny2','Sunny3','Sunny4']


# * This function extracts decaf features in batches of a list of images

# In[ ]:

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


# * Finally, we call the function

# In[ ]:
print("AlexNet")
for k in illumination:
    print(k)
    list_images = os.listdir(images_root % k)
    list_images.sort()
    features_path = "features/KTH-IDOL/Minnie/%s/" % k
    
    if not os.path.exists(features_path):
        os.makedirs(features_path)
        
    if not os.path.exists(features_path+"pool5_caffenet"):
        os.makedirs(features_path+"pool5_caffenet")
    
    if not os.path.exists(features_path+"fc6_caffenet"):
        os.makedirs(features_path+"fc6_caffenet")
    
    if not os.path.exists(features_path+"fc7_caffenet"):
        os.makedirs(features_path+"fc7_caffenet")    
    
    decaf_features = extract_decaf_features(images_root%k,list_images,'pool5',net,transformer)
    pickle.dump(decaf_features, gzip.open( features_path+"pool5_caffenet/pool5_caffenet.pklz", "wb" ), 2)

    decaf_features = extract_decaf_features(images_root%k,list_images,'fc6',net,transformer)
    pickle.dump(decaf_features, gzip.open( features_path+"fc6_caffenet/fc6_caffenet.pklz", "wb" ), 2)

    decaf_features = extract_decaf_features(images_root%k,list_images,'fc7',net,transformer)
    pickle.dump(decaf_features, gzip.open( features_path+"fc7_caffenet/fc7_caffenet.pklz", "wb" ), 2)

# In[ ]:

model_def = caffe_root + 'models/ResNet-152/ResNet-152-deploy.prototxt'
model_weights = caffe_root + 'models/ResNet-152/ResNet-152-model.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# In[ ]:

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# In[ ]:
print()
print("ResNet-152")
for k in illumination:
    print(k)
    list_images = os.listdir(images_root % k)
    list_images.sort()
    
    features_path = "features/KTH-IDOL/Minnie/%s/" % k

    if not os.path.exists(features_path+"pool5_ResNet-152"):
        os.makedirs(features_path+"pool5_ResNet-152")

    decaf_features = extract_decaf_features(images_root%k,list_images,'pool5',net,transformer)
    pickle.dump(decaf_features, gzip.open( features_path+"pool5_ResNet-152/pool5_ResNet-152.pklz", "wb" ), 2)

