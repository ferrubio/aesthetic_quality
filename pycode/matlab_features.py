
# coding: utf-8

# # Classifiers based on features extracted from matlab
#
# In this notebook we use the different arff files obtained from matlab. We will use this features to obtain classifiers and test them in a cross validation process.
#
# ## A bit of set up
#
# We need numpy and pandas for data. Pickle and gzip for read the extracted features

# In[ ]:

# set up Python environment: numpy for numerical routines
import numpy as np
import pandas as pd

# for store the results
import pickle
import gzip


# In this example we only use the default linear SVM classifier from libsvm and the Gaussian NB from sklearn

# In[ ]:

from sklearn import svm
from sklearn.naive_bayes import GaussianNB


# In[ ]:

import sys
sys.path.append('/home/frubio/python/mlframework/mlframework/')
sys.path.append('pycode/')
from dataset.normal_dataset import normal_dataset
import utilsData


# ## Feature reading
#
# In this example only one package is read, but each ones have a size of 80Mb approximately.

# In[ ]:

data = pickle.load(gzip.open('packages/info.pklz','rb',2))


# In[ ]:

total_files=['PHOG/','CHIST.arff', 'GHIST.arff', 'GIST_ori8_block4.arff', 'Centrist.arff']
phog_files=['2_bins360_levels0_angle360.arff', '3_bins300_levels0_angle360.arff', '4_bins200_levels0_angle360.arff',
            '5_bins100_levels0_angle360.arff', '6_bins50_levels0_angle360.arff', '7_bins20_levels0_angle360.arff',
            '8_bins100_levels1_angle360.arff', '9_bins50_levels1_angle360.arff', '10_bins20_levels1_angle360_redux.arff',
            '11_bins50_levels2_angle360.arff', '12_bins20_levels2_angle360.arff']

delta = int(sys.argv[3])
# this is for use the same random than in DeCAF
batches = 100


# In[ ]:

if sys.argv[1] == '0':
    features = utilsData.readARFF('features/AVA/'+total_files[int(sys.argv[1])]+phog_files[int(sys.argv[2])])
else:
    features = utilsData.readARFF('features/AVA/'+total_files[int(sys.argv[1])])

# In[ ]:

features=pd.DataFrame(features['data'],columns=features['vars'])
features=features.rename(columns=(lambda x: 'var'+str(int(x[3:])-1)))
features=features.rename(columns={'var0':'id'})

num_features = len(features.columns)-1
data=pd.merge(data, features, on='id', how='right')


# In[ ]:

num_images = len(data)

votesList=np.array(data.iloc[:,2:12])
auxTotal=np.sum(votesList,axis=1)
auxMeanVector=np.sum(votesList*range(1,11),axis=1)/auxTotal.astype(np.float)
auxClass=np.array(auxMeanVector >= 5, dtype=np.int)

data.loc[:,'VotesMean'] = pd.Series(auxMeanVector, index=data.index)
data.loc[:,'Class'] = pd.Series(auxClass, index=data.index)


# In[ ]:

data.loc[:,'id'] = data['id'].apply(str)
classes = np.array(data.sort_values(['id']).loc[:,'Class'])
features = np.array(data.sort_values(['id']).iloc[:,37:num_features+37])
means = np.array(data.sort_values(['id']).loc[:,'VotesMean'])


# In[ ]:

indexes = np.array(range(0,len(classes))[:len(classes)-(len(classes) % batches)])
indexes = indexes.reshape((batches,-1))


# ## Cross validation
#
# In this case, we prepare vectors with the batches of each fold in order to test them in galgo and store the results.
#
# * First, we split the batches in 5 folds:

# In[ ]:

np.random.seed(1000)
num_folds = 5
folds = np.random.choice(range(0,batches),replace=False,size=(num_folds,batches/num_folds))


# In[ ]:

def balance_class(features, classes):
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

    return (features[final_indexes],classes[final_indexes])



# In[ ]:

sum_folds_svm = 0
sum_folds_nbg = 0
matrix_svm = np.zeros((2,2))
matrix_nbg = np.zeros((2,2))
for i in range(0, num_folds):

    # Prepare train
    train_indices = indexes[np.delete(folds,i,axis=0).reshape(-1)].reshape(-1)
    train_features = features[train_indices]
    train_classes = classes[train_indices]
    train_means = means[train_indices]

    # Delete values depending on the delta
    vector_out_delta = (train_means <= 5-delta) | (train_means >= 5+delta)
    train_features = train_features[vector_out_delta]
    train_classes = train_classes[vector_out_delta]

    # Class balance
    train_features,train_classes = balance_class(train_features,train_classes)

    # Fit models
    svm_clf = svm.LinearSVC()
    svm_clf.fit(train_features, train_classes)

    nbg_clf = GaussianNB()
    nbg_clf.fit(train_features, train_classes)

    # Prepare test
    test_indices = indexes[folds[i]].reshape(-1)
    test_features = features[test_indices]
    test_classes = classes[test_indices]

    # Evaluate SVM model
    predictions = svm_clf.predict(test_features)
    results = np.sum(predictions == test_classes)/float(len(predictions))
    sum_folds_svm += results

    matrix_svm[0,0] += np.sum(predictions[predictions == test_classes] == 0)
    matrix_svm[0,1] += np.sum(predictions[predictions != test_classes] == 1)
    matrix_svm[1,0] += np.sum(predictions[predictions != test_classes] == 0)
    matrix_svm[1,1] += np.sum(predictions[predictions == test_classes] == 1)

    # Evaluate gnb model
    predictions = nbg_clf.predict(test_features)
    results = np.sum(predictions == test_classes)/float(len(predictions))
    sum_folds_nbg += results

    matrix_nbg[0,0] += np.sum(predictions[predictions == test_classes] == 0)
    matrix_nbg[0,1] += np.sum(predictions[predictions != test_classes] == 1)
    matrix_nbg[1,0] += np.sum(predictions[predictions != test_classes] == 0)
    matrix_nbg[1,1] += np.sum(predictions[predictions == test_classes] == 1)


# In[ ]:

data_results = {'accuracy':sum_folds_svm/num_folds, 'conf_matrix':matrix_svm, 'classifier':'SVM-L', 'descriptor':total_files[int(sys.argv[1])], 'delta':delta}
if sys.argv[1] == '0':
    data_results['case']=phog_files[int(sys.argv[2])]
    pickle.dump(data_results, gzip.open("results/SVM_balanced_Descriptor%d_Case%d.pklz" % (int(sys.argv[1]),int(sys.argv[2])), "wb" ), 2)
else:
    pickle.dump(data_results, gzip.open("results/SVM_balanced_Descriptor%d.pklz" % (int(sys.argv[1])), "wb" ), 2)

data_results = {'accuracy':sum_folds_nbg/num_folds, 'conf_matrix':matrix_nbg, 'classifier':'NB-G', 'descriptor':total_files[int(sys.argv[1])], 'delta':delta}
if sys.argv[1] == '0':
    data_results['case']=phog_files[int(sys.argv[2])]
    pickle.dump(data_results, gzip.open("results/GNB_balanced_Descriptor%d_Case%d.pklz" % (int(sys.argv[1]),int(sys.argv[2])), "wb" ), 2)
else:
    pickle.dump(data_results, gzip.open("results/GNB_balanced_Descriptor%d.pklz" % (int(sys.argv[1])), "wb" ), 2)
