# set up Python environment: numpy for numerical routines
import numpy as np
import pandas as pd

# for store the results
from six.moves import cPickle as pickle
import gzip

# our code (utilsData needs a view)
import sys
sys.path.append('pycode/')
import utilsData

from sklearn.metrics import roc_auc_score, accuracy_score
import full_models
from preprocess import utilities

import sqlite3

features_file = sys.argv[1]
output_file = sys.argv[2]
selected_model = sys.argv[3]
decaf_discrete = sys.argv[4]

if features_file[-4:] == 'pklz':
    features = pickle.load(open(features_file,'rb',pickle.HIGHEST_PROTOCOL))
else:
    features = utilsData.readARFF(features_file)
    
features['id'] = features['id'].astype(int)

# we take the name of the features and delete de ID
features_names = np.array(features.columns)
index = np.argwhere(features_names=='id')
features_names = np.delete(features_names, index)

# this line is for normalize decaf features
if (decaf_discrete == 'True'):
    features[features_names],_ = utilities.reference_forward_implementation(np.array(features[features_names]),5,2,1.5,0.75)

data = pickle.load(gzip.open('packages/AVA_info.pklz','rb',2))
data = data.merge(features, on='id', copy=False)

del features

num_images = data.shape[0]

data_aux = data[np.append(features_names,['Class'])]
data_aux['Class'] = pd.Categorical(data_aux['Class'],range(0,len(data_aux['Class'].unique())))

# and free space
del data

train_indices = pickle.load(gzip.open('../models/train_indexes_AesNet.pklz','rb',2))
test_indices = pickle.load(gzip.open('../models/test_indexes_AesNet.pklz','rb',2))

if selected_model == 'NB':
    predictions = full_models.fullNB(data_aux, train_indices, test_indices)

elif selected_model == 'AODE':
    predictions = full_models.fullAODE(data_aux, train_indices, test_indices)

elif selected_model == 'NBG':
    predictions = full_models.fullNBG(data_aux, train_indices, test_indices, features_names, 'Class')

elif selected_model == 'SVM':
    predictions = full_models.fullSVM(data_aux, train_indices, test_indices, features_names, 'Class')

elif selected_model == 'ELM':
    predictions = full_models.fullELM(data_aux, train_indices, test_indices, features_names, 'Class')

elif selected_model == 'GBoost':
    predictions = full_models.fullGBoost(data_aux, train_indices, test_indices, features_names, 'Class')
    
results = {}
results['balanced'] = utilsData.balanced_accuracy(data_aux['Class'].cat.codes[test_indices], predictions)
results['AUC'] = roc_auc_score(data_aux['Class'].cat.codes[test_indices], predictions)
results['accuracy'] = accuracy_score(data_aux['Class'].cat.codes[test_indices], (predictions >= 0.5).astype(int))

pickle.dump(results, gzip.open( output_file, "wb" ), 2)
