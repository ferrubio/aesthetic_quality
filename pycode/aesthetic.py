# set up Python environment: numpy for numerical routines
import numpy as np
import pandas as pd

# for store the results
import pickle
import gzip

# our code (utilsData needs a view)
import sys
sys.path.append('pycode/')
import utilsData

from sklearn.metrics import roc_auc_score, accuracy_score
import full_models

features = utilsData.readARFF(sys.argv[1])
output_file = sys.argv[2]
selected_model = sys.argv[3]


data = pickle.load(gzip.open('packages/AVA_info.pklz','rb',2))

# we take the name of the features and delete de ID
features_names = np.array(features.columns)
index = np.argwhere(features_names=='id')
features_names = np.delete(features_names, index)

data=pd.merge(data, features, on='id', how='right')
num_images = data.shape[0]

# to free space
del features

data_aux = data[np.append(features_names,['Class'])]
data_aux['Class'] = pd.Categorical(data_aux['Class'],data_aux['Class'].unique())

np.random.seed(1000)
num_folds = 5
folds = np.random.choice(range(0,num_images),replace=False,size=(num_folds,int(num_images/num_folds)))

results = {}
results['balanced']=0
results['AUC']=0
results['accuracy']=0

for i in range(0, num_folds):
    
    train_indices = np.delete(folds,i,axis=0).reshape(-1)
    train_indices = train_indices[utilsData.balance_class(data_aux['Class'].cat.codes[train_indices])]
    
    test_indices = folds[i]
    
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
   
    results['balanced'] += utilsData.balanced_accuracy(data_aux['Class'].cat.codes[test_indices], predictions)
    results['AUC'] += roc_auc_score(data_aux['Class'].cat.codes[test_indices], predictions)
    results['accuracy'] += accuracy_score(data_aux['Class'].cat.codes[test_indices], (predictions >= 0.5).astype(int))
    
results['balanced'] /= num_folds
results['AUC'] /= num_folds
results['accuracy'] /= num_folds

pickle.dump(results, gzip.open( output_file, "wb" ), 2)
