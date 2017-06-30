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


import aestheticNet
from preprocess import utilities 

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

niter = 10  # number of iterations to train

# Reset style_solver as before.
style_solver_filename = aestheticNet.solver(aestheticNet.aest_net(train=True,caffe_aes=True))
style_solver = caffe.get_solver(style_solver_filename)
style_solver.net.copy_from(weights)

print ('Running solvers for %d iterations...' % niter)
solvers = [('pretrained', style_solver)]
loss, acc, weights = aestheticNet.run_solvers(niter, solvers)
print ('Done.')

train_loss = loss['pretrained']
train_acc = acc['pretrained']
style_weights = weights['pretrained']

# Delete solvers to save memory.
del style_solver, solvers

os.rename(weights['pretrained'], "/home/frubio/AVA/aesthetic_finetuning.caffemodel")