
# coding: utf-8

# ## A bit of set up
# 
# We need numpy and pandas for data. Pickle and gzip for read the extracted features

# In[1]:

# set up Python environment: numpy for numerical routines
import numpy as np
import pandas as pd

# for store the results
import pickle
import gzip


# In this example we only use the default linear SVM classifier (SVC) from sklearn

# In[2]:

import sys
sys.path.append('/home/frubio/Mat_Docs/libsvm-3.20/python/')

from svmutil import *
from sklearn.naive_bayes import GaussianNB


# ## Feature reading
# 
# In this example only one package is read, but each ones have a size of 80Mb approximately.

# In[25]:

directory_file = "fc6_caffenet/fc6_caffenet_%02d.pklz"
batches = 100


# In[27]:

features=pickle.load(gzip.open(directory_file % 0,'rb',2))


# In[28]:

batch_H = features.shape[0]
features = features.reshape((batch_H,-1))
batch_W = features.shape[1]


# Now is the turn for the classes:
# * First we read the information of AVA in pandas dataframe format

# In[12]:

data=pickle.load(gzip.open('../info.pklz','rb',2))


# * We calculate the mean of the votes and the weight (class)

# In[13]:

num_images=len(data)
auxWeight=np.zeros(num_images,dtype=np.int)
auxMeanVector=np.zeros(num_images, dtype=np.double)
votesList=np.array(data.iloc[:,2:12])
auxTotal=np.sum(votesList,axis=1)
auxMeanVector=np.sum(votesList*range(1,11),axis=1)/auxTotal.astype(np.float)
auxWeight=np.array(auxMeanVector>=5, dtype=np.int)

# for initial class 1 or 0
data.loc[:,'Weight'] = pd.Series(auxWeight, index=data.index)
data.loc[:,'VotesMean'] = pd.Series(auxMeanVector, index=data.index)


# * Finally, we transform the id to string and sort the information to extract the corresponding classes in a vector

# In[14]:

data.loc[:,'id'] = data['id'].apply(str)
classes = np.array(data.sort_values(['id']).loc[:,'Weight'])


# * In order to have the same structure with respect to the features, where they are splitted in batches, we do the same with the classes

# In[15]:

classes = classes[:len(classes)-(len(classes) % batches)]
classes = classes.reshape((batches,-1))

# ## Cross validation
# 
# In this case, we prepare vectors with the batches of each fold in order to test them in galgo and store the results.
# 
# * First, we split the batches in 5 folds:

# In[16]:

np.random.seed(1000)
num_folds = 5
folds = np.random.choice(range(0,batches),replace=False,size=(num_folds,batches/num_folds))


# * We start the for, where the features are read and resimensioned in order to train the model, and then, the test is read in the same way and the predictions are made

# In[21]:

def read_and_format_features(indices_list,batch_H,batch_W,directory_file):
    num_batches = len(indices_list)
    features = np.zeros((num_batches*batch_H, batch_W))
    
    pre_count = 0
    post_count = batch_H
    
    for i in indices_list:
        features_aux = pickle.load(gzip.open(directory_file % i,'rb',2))
        features[pre_count:post_count] = features_aux.reshape((batch_H, batch_W))
        pre_count = post_count
        post_count += batch_H 
        
    return features
    


# In[18]:

def read_and_format_classes(indices_list, batch_H, classes):
    num_batches = len(indices_list)
    train_classes = np.zeros(num_batches*batch_H)
    
    pre_count = 0
    post_count = batch_H

    for i in indices_list:
        train_classes[pre_count:post_count] = classes[i]
        
        pre_count = post_count
        post_count += batch_H 
        
    return train_classes
    


# In[ ]:

results = np.zeros(num_folds)
for i in range(0, num_folds):
    
    # Prepare train
    train_indices = np.delete(folds,i,axis=0).reshape(-1)
    
    features = read_and_format_features(train_indices[0:1],batch_H,batch_W,directory_file)
    train_classes = read_and_format_classes(train_indices[0:1],batch_H,classes)
    
    # Fit models
    prob  = svm_problem(train_classes.tolist(), features.tolist())
    param = svm_parameter('-t 0')
    svm_clf = svm_train(prob, param)

    gnb_clf = GaussianNB()
    gnb_clf.fit(features, train_classes)
    
    # Prepare test
    test_indices = folds[i]
    
    features = read_and_format_features(test_indices[0:1],batch_H,batch_W,directory_file)
    test_classes = read_and_format_classes(test_indices[0:1],batch_H,classes)
    
    # Evaluate model
    _, p_acc, _ = svm_predict(test_classes.tolist(), features.tolist(), svm_clf)
    pickle.dump(p_acc[0]/100, gzip.open( "results/SVM_fc6_fold%d.pklz" % i, "wb" ), 2)
    
    predictions = gnb_clf.predict(features)
    results = np.sum(predictions == test_classes)/float(len(predictions))
    pickle.dump(results, gzip.open( "results/GNB_fc6_fold%d.pklz" % i, "wb" ), 2)
    

