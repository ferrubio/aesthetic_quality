
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


# In this example we only use the default linear SVM classifier from libsvm and the Gaussian NB from sklearn

# In[2]:

import sys
sys.path.append('/home/frubio/Mat_Docs/libsvm-3.20/python/')

from svmutil import *
from sklearn.naive_bayes import GaussianNB


# In[3]:

sys.path.append('/home/frubio/PycharmProjects/mlframework/mlframework/')
from dataset.normal_dataset import normal_dataset
import utilsData


# ## Feature reading
# 
# In this example only one package is read, but each ones have a size of 80Mb approximately.

# In[4]:

data = pickle.load(gzip.open('/home/frubio/python/EMtest/info.pklz','rb',2))


# In[5]:

main_path='/home/frubio/EM_descriptors/'
total_files=['PHOG/','CHIST.arff', 'GHIST.arff', 'GIST_ori8_block4.arff', 'Centrist.arff']
phog_files=['2_bins360_levels0_angle360.arff', '3_bins300_levels0_angle360.arff', '4_bins200_levels0_angle360.arff',
            '5_bins100_levels0_angle360.arff', '6_bins50_levels0_angle360.arff', '7_bins20_levels0_angle360.arff',
            '8_bins100_levels1_angle360.arff', '9_bins50_levels1_angle360.arff', '10_bins20_levels1_angle360_redux.arff',
            '11_bins50_levels2_angle360.arff', '12_bins20_levels2_angle360.arff']


# In[ ]:

if sys.argv[1] == '0':
    features = utilsData.readARFF(main_path+'AVA/'+total_files[int(sys.argv[1])]+phog_files[int(sys.argv[2])])
else:
    features = utilsData.readARFF(main_path+'AVA/'+total_files[int(sys.argv[1])])


# In[7]:

features=pd.DataFrame(features['data'],columns=features['vars'])
features=features.rename(columns=(lambda x: 'var'+str(int(x[3:])-1)))
features=features.rename(columns={'var0':'id'})

num_features = len(features.columns)-1
data=pd.merge(data, features, on='id', how='right')


# In[8]:

num_images = len(data)

votesList=np.array(data.iloc[:,2:12])
auxTotal=np.sum(votesList,axis=1)
auxMeanVector=np.sum(votesList*range(1,11),axis=1)/auxTotal.astype(np.float)
auxWeight=np.array(auxMeanVector >= 5, dtype=np.int)

data.loc[:,'VotesMean'] = pd.Series(auxMeanVector, index=data.index)
data.loc[:,'Weight'] = pd.Series(auxWeight, index=data.index)


# In[9]:

data.loc[:,'id'] = data['id'].apply(str)
classes = np.array(data.sort_values(['id']).loc[:,'Weight'])
features = np.array(data.sort_values(['id']).iloc[:,37:num_features+37])


# In order to use the same cases than in DeCAFF we have to separate the features and the classes in batches

# In[10]:

batches = 10


# In[11]:

classes = classes[:len(classes)-(len(classes) % batches)]
classes = classes.reshape((batches,-1))


# In[12]:

features = features[:len(features)-(len(features) % batches)]
features = features.reshape((batches,-1,num_features))


# ## Cross validation
# 
# In this case, we prepare vectors with the batches of each fold in order to test them in galgo and store the results.
# 
# * First, we split the batches in 5 folds:

# In[13]:

np.random.seed(1000)
num_folds = 5
folds = np.random.choice(range(0,batches),replace=False,size=(num_folds,batches/num_folds))


# In[ ]:

for i in range(0, num_folds):
    
    # Prepare train
    train_indices = np.delete(folds,i,axis=0).reshape(-1)
    
    train_features = features[train_indices].reshape((-1,num_features))
    train_classes = classes[train_indices].reshape((-1))
    
    # Fit models
    prob  = svm_problem(train_classes.tolist(), train_features.tolist())
    param = svm_parameter('-t 0')
    svm_clf = svm_train(prob, param)

    gnb_clf = GaussianNB()
    gnb_clf.fit(train_features, train_classes)
    
    # Prepare test
    test_indices = folds[i]
    
    test_features = features[test_indices].reshape((-1,num_features))
    test_classes = classes[test_indices].reshape((-1))
    
    # Evaluate model
    _, p_acc, _ = svm_predict(test_classes.tolist(), test_features.tolist(), svm_clf)
    if sys.argv[1] == '0':
        pickle.dump(p_acc[0]/100, gzip.open( "results/SVM_Descriptor%d_Case%d_fold%d.pklz" % (int(sys.argv[1]),int(sys.argv[2]),i), "wb" ), 2)
    else:
        pickle.dump(p_acc[0]/100, gzip.open( "results/SVM_Descriptor%d_fold%d.pklz" % (int(sys.argv[1]),i), "wb" ), 2)
    
    predictions = gnb_clf.predict(test_features)
    results = np.sum(predictions == test_classes)/float(len(predictions))
    if sys.argv[1] == '0':
        pickle.dump(results, gzip.open( "results/GNB_Descriptor%d_Case%d_fold%d.pklz" % (int(sys.argv[1]),int(sys.argv[2]),i), "wb" ), 2)
    else:
        pickle.dump(results, gzip.open( "results/GNB_Descriptor%d_fold%d.pklz" % (int(sys.argv[1]),i), "wb" ), 2)

