
# coding: utf-8

# In[1]:

import os.path
import numpy as np
import cv2
import sys
import pickle
import gzip

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

mainPath='/home/frubio/AVA/'


# In[2]:

def extractSizeFromRadius(radius):
    SIFT_DESCR_WIDTH = 4
    SIFT_DESCR_SCL_FCTR = 3.0

    final_size = radius / (SIFT_DESCR_SCL_FCTR * (SIFT_DESCR_WIDTH + 1) * 0.5 * 0.5 * 1.4142135623730951)
    return final_size


# In[3]:

def extractdSIFTatScales(file, scales, initial_radius, factor_step):
    img = cv2.imread(mainPath+'AVADataset/'+file+'.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    radius = initial_radius

    for i in range(0,scales):
        step_size = extractSizeFromRadius(radius)
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(radius, gray.shape[0], radius*2)
                                        for x in range(radius, gray.shape[1], radius*2)]
        dense_feat = sift.compute(gray, kp)

        if i==0:
            final_feat = dense_feat[1]
        else:
            final_feat = np.concatenate((final_feat,dense_feat[1]), axis=0)

        radius = int(np.around(radius*factor_step))

    pickle.dump(final_feat, gzip.open("/home/frubio/aesthetic_quality/features/dSIFT/initialRad{:d}_scales{:d}_factor{:.1f}/AVA/{}.pklz".format(initial_radius,scales,factor_step,file), "wb" ),2)
    #pickle.dump(final_feat, gzip.open("/home/frubio/aesthetic_quality/features/dSIFT/{}.pklz".format(file), "wb" ),2)


# In[4]:

scales = 5
initial_radius = 16
factor_step = 1.2

f = open(mainPath+'AVA_files/AVA.txt', 'r')

for line in f:
    strSplit=line.split()
    fname=mainPath+'AVADataset/'+strSplit[1]+'.jpg'
    if os.path.isfile(fname):
        extractdSIFTatScales(strSplit[1], scales, initial_radius, factor_step)

f.close()
