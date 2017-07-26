# own models and functions
from preprocess.mdl import MDL_method
from preprocess.unsupervised import Unsupervised_method
from models.nb import Naive_Bayes
from models.aode_fast import AODE_fast

# default models from scikit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn_extensions.extreme_learning_machines.random_layer import MLPRandomLayer
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn.ensemble import GradientBoostingClassifier


def fullNB(data_aux, train_indices, test_indices):
    discretization = MDL_method()
    # discretization.frequency = True
    # discretization.bins = 5
    discretization.train(data_aux.loc[train_indices])
    data_fold = discretization.process(data_aux)

    model = Naive_Bayes()
    model.fit(data_fold.loc[train_indices])

    return model.predict_probs(data_fold.loc[test_indices])[1]


def fullAODE(data_aux, train_indices, test_indices):
    discretization = MDL_method()
    #discretization.frequency = True
    #discretization.bins = 5
    discretization.train(data_aux.loc[train_indices])
    data_fold = discretization.process(data_aux)

    model = AODE_fast()
    model.fit(data_fold.loc[train_indices])

    return model.predict_probs(data_fold.loc[test_indices])[:,1]


def fullNBG(data_aux, train_indices, test_indices, features_names, class_name):
    data_fold = data_aux.copy()

    model = GaussianNB()
    model.fit(data_fold.loc[train_indices,features_names],data_fold[class_name].cat.codes[train_indices])

    return model.predict_proba(data_fold.loc[test_indices,features_names])[:,1]
    
    
def fullSVM(data_aux, train_indices, test_indices, features_names, class_name):
    data_fold = data_aux.copy()

    model = LinearSVC()
    model.fit(data_fold.loc[train_indices,features_names],data_fold[class_name].cat.codes[train_indices])

    return model.predict(data_fold.loc[test_indices,features_names])
    
    
def fullELM(data_aux, train_indices, test_indices, features_names, class_name):
    data_fold = data_aux.copy()

    nh = 1000
    srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    model = GenELMClassifier(hidden_layer=srhl_tanh)
    model.fit(data_fold.loc[train_indices,features_names],data_fold[class_name].cat.codes[train_indices])
    
    return model.predict(data_fold.loc[test_indices,features_names])


def fullGBoost(data_aux, train_indices, test_indices, features_names, class_name):
    data_fold = data_aux.copy()

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0) 
    model.fit(data_fold.loc[train_indices,features_names],data_fold[class_name].cat.codes[train_indices])
    
    return model.predict(data_fold.loc[test_indices,features_names])

