
# coding: utf-8

# In[1]:

import os.path
import numpy as np
import cv2
import sys
import pickle
import gzip

mainPath='/home/frubio/AVA/'


# In[ ]:

def extractSIFT(file, step_size):
    img = cv2.imread(mainPath+'AVADataset/'+file+'.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size) 
                                    for x in range(0, gray.shape[1], step_size)]
    
    sift = cv2.xfeatures2d.SIFT_create()
    dense_feat = sift.compute(gray, kp)
    pickle.dump(dense_feat[1], gzip.open("/home/frubio/aesthetic_quality/features/dSIFT/AVA/"+file+".pklz", "wb" ),2)


# In[2]:

step_size = 16

f = open(mainPath+'AVA_files/AVA.txt', 'r')

for line in f:
    strSplit=line.split()
    fname=mainPath+'AVADataset/'+strSplit[1]+'.jpg'
    if os.path.isfile(fname):
        extractSIFT(strSplit[1], step_size)
        
f.close()

