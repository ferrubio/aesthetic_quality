# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
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


import AesNet
from preprocess import utilities 

caffe.set_device(0)  # if we have multiple GPUs, pick the first one

# We select the model
weights = caffe_root + 'models/VGG-16/VGG_ILSVRC_16_layers.caffemodel'
#weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

# The fully connected nodes
fc_nodes = 500

# The partition used
# partition_pref = '' # (66% train, 33% test and balanced train (90,000 train images))
#partition_pref = '_standard_AVA' # (92% train, 8% test (200,000 train images))
partition_pref = '_standard_AVA_balanced' # (92% train, 8% test and balanced train (130,000 train images))


model_pref = 'AesNet_VGG16_{}{}'.format(fc_nodes,partition_pref)
#model_pref = AesNet_CaffeNet_{}{}.format(fc_nodes,partition_pref)

niter = 30000  # number of iterations to train

# Reset style_solver as before.
style_solver_filename = AesNet.solver(AesNet.AesNet_VGG16(train=True, 
                                                    source_path='models/%s_partition_finetuning{}.txt'.format(partition_pref),
                                                            fc_nodes=fc_nodes),
                                           snapshot_pref = model_pref,
                                           base_lr=0.001)
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(weights)

print ('Running solvers for %d iterations...' % niter)
solvers = [('pretrained', style_solver)]
loss, acc, weights = AesNet.run_solvers(niter, solvers)
print ('Done.')

train_loss = loss['pretrained']
train_acc = acc['pretrained']
style_weights = weights['pretrained']

# Delete solvers to save memory.
del style_solver, solvers

os.rename(weights['pretrained'], model_pref+".caffemodel")
