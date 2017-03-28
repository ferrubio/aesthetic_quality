
# coding: utf-8

# In[1]:

import os, os.path
import numpy as np
import pandas as pd

import sys
import pickle
import gzip
import sys

mainPath='/home/frubio/AVA/'
featuresPath = "/home/frubio/python/aesthetic_quality/features/dSIFT/initialRad{:d}_scales{:d}_factor{:.1f}/AVA/"

sys.path.insert(0,'pycode')
import fisher_vector
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import SGDClassifier

# In[2]:

# Parameters of the classification
delta = 0

# Parameters of the descriptors
scales = 5
initial_radius = 16
factor_step = 1.2

# Parameters for the FV
size_patch = 500000
size_PCA = 64
size_gmm = 128

# Parameters for the cross validation
np.random.seed(1000)
batches = 100
num_folds = 5
folds = np.random.choice(range(0,batches),replace=False,size=(num_folds,int(batches/num_folds)))


# In[3]:

data = pickle.load(gzip.open('packages/info.pklz','rb',2))
num_images = len(data)

votesList=np.array(data.iloc[:,2:12])
auxTotal=np.sum(votesList,axis=1)
auxMeanVector=np.sum(votesList*range(1,11),axis=1)/auxTotal.astype(np.float)
auxClass=np.array(auxMeanVector >= 5, dtype=np.int)

data.loc[:,'VotesMean'] = pd.Series(auxMeanVector, index=data.index)
data.loc[:,'Class'] = pd.Series(auxClass, index=data.index)


# In[4]:

data.loc[:,'id'] = data['id'].apply(str)
classes = np.array(data.sort_values(['id']).loc[:,'Class'])
means = np.array(data.sort_values(['id']).loc[:,'VotesMean'])


# In[5]:

indexes = np.array(range(0,len(classes))[:len(classes)-(len(classes) % batches)])
indexes = indexes.reshape((batches,-1))


# In[6]:

def balance_class(indexes, classes):
    classes_uniques = np.unique(classes)
    min_class = np.array([0,float('Inf')])
    for i in classes_uniques:
        aux_value = np.sum(classes == i)
        if aux_value < min_class[1]:
            min_class = np.array([i,aux_value])
            
    final_indexes = np.where(classes == min_class[0])[0]
    for i in classes_uniques:
        if i != min_class[0]:
            aux_indexes = np.where(classes == i)[0]
            #print np.random.choice(aux_indexes,replace=False,size=min_class[1])
            final_indexes = np.concatenate((final_indexes,np.random.choice(aux_indexes,replace=False,size=min_class[1])))
            
    final_indexes = np.sort(final_indexes)
    
    return (final_indexes,classes[final_indexes])


# In[ ]:

class FV_dictionary:
    
    def __init__(self,size_PCA, size_patch, size_gmm):
        self.size_patch = size_patch
        self.pca = PCA(n_components=size_PCA)
        self.gmm = GaussianMixture(n_components=size_gmm, covariance_type='diag')
        self.size_descriptor = 0
        
    def generate_dict(self,indexes,files,path):
        
        matrix_features = self.extract_patch_features(indexes, files, path)
        
        descriptor_size = matrix_features.shape[1]
        if descriptor_size > self.pca.n_components:
            self.pca.fit(matrix_features)
            matrix_features = self.pca.transform(matrix_features)
            self.size_descriptor = self.pca.n_components
        else:
            self.size_descriptor = descriptor_size
            
        self.gmm.fit(matrix_features)
        
    def obtain_fv(self,indexes,files,path):
        
        fv_size = self.gmm.n_components*(1+2*self.size_descriptor)
        final_matrix = np.zeros((indexes.shape[0],fv_size))
        counter = 0
        for i in indexes:
            fname=path+files[i]+'.pklz'
            if os.path.isfile(fname):
                sift = pickle.load(gzip.open(fname,"rb",2))
                descriptor_size = sift.shape[1]
                if descriptor_size > self.pca.n_components:
                    sift = self.pca.transform(sift)
                final_matrix[counter] = fisher_vector.fisher_vector(sift, self.gmm)
                counter += 1
        return final_matrix
    
    def extract_patch_features(self, indexes, files, path):
        # We extract the number of vectors corresponding to the size of the patch / number of images
        nImages = indexes.shape[0]
        featuresPerImage = int(self.size_patch / nImages)
        finalMatrix = np.zeros((featuresPerImage*nImages, 128),dtype=np.float32)

        counter = 0
        
        for i in indexes:
            fname=path+files[i]+'.pklz'
            if os.path.isfile(fname):
                sift = pickle.load(gzip.open(fname,"rb",2))
                selectedFeat = np.random.choice(range(0,sift.shape[0]),replace=False,size=featuresPerImage)
                finalMatrix[counter:counter+featuresPerImage] = sift[selectedFeat]
            counter += featuresPerImage
        return finalMatrix


# In[ ]:

sum_folds_sgd = 0
matrix_sgd = np.zeros((2,2))

for i in range(0, num_folds):
    print ("Fold {:d}".format(i))
    # Prepare train
    train_indexes = indexes[np.delete(folds,i,axis=0).reshape(-1)].reshape(-1)
    train_means = means[train_indexes]
    
    # Delete values depending on the delta
    vector_out_delta = (train_means <= 5-delta) | (train_means >= 5+delta)
    train_indexes = train_indexes[vector_out_delta]
    train_classes = classes[train_indexes]
    
    # Class balance
    train_indexes,train_classes = balance_class(train_indexes,train_classes)
    

    # Only take into account those features from the final train set
    dictionary = FV_dictionary(size_PCA,size_patch,size_gmm)
    dictionary.generate_dict(train_indexes,np.array(data.sort_values(['id'])['id']),featuresPath.format(initial_radius,scales,factor_step))
    train_features = dictionary.obtain_fv(train_indexes,np.array(data.sort_values(['id'])['id']),featuresPath.format(initial_radius,scales,factor_step))
    print("features shape")
    print(train_features.shape)
    
    # Fit models
    sgd_clf = SGDClassifier(loss="hinge", penalty="l2")
    sgd_clf.fit(train_features, train_classes)
    
    # Prepare test
    test_indexes = indexes[folds[i]].reshape(-1)
    test_features = dictionary.obtain_fv(test_indexes,np.array(data.sort_values(['id'])['id']),featuresPath.format(initial_radius,scales,factor_step))
    test_classes = classes[test_indexes]
    print("test shape")
    print(test_features.shape)
    
    # Evaluate SVM model
    predictions = sgd_clf.predict(test_features)
    results = np.sum(predictions == test_classes)/float(len(predictions))
    sum_folds_sgd += results
    
    matrix_sgd[0,0] += np.sum(predictions[predictions == test_classes] == 0)
    matrix_sgd[0,1] += np.sum(predictions[predictions != test_classes] == 1)
    matrix_sgd[1,0] += np.sum(predictions[predictions != test_classes] == 0)
    matrix_sgd[1,1] += np.sum(predictions[predictions == test_classes] == 1)
    
# In[ ]:

data_results = {'accuracy':sum_folds_sgd/num_folds, 'conf_matrix':matrix_sgd, 'classifier':'SGD', 'descriptor':'SIFT+FV', 'delta':delta}
pickle.dump(data_results, gzip.open("results/SGD_balanced_DescriptorFV_delta{:.1f}.pklz".format(delta), "wb" ), 2)
