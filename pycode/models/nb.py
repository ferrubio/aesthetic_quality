# -*- coding: utf-8 -*-

"""
This is the first vesrion of Naive Bayes module. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

# Author: Fernando Rubio <fernando.rubio@uclm.es>
#

import numpy as np
import pandas as pd
import warnings
from scipy.misc import logsumexp


class Naive_Bayes():

    '''
    ================================================
    == Parameters file =============================
    ================================================
    <model>
        <module>bayes</module>
        <name>naive_bayes</name>
        <params>
            <alpha>1.0</alpha>
            <fit_prior>True</fit_prior>
            <class_prior>None</class_prior>
        </params>
    </model>

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    '''

    def __init__(self):
        
        # Specific input parameters
        self.class_index = None
        self.alpha = 1
        self.fit_prior = True
        self.class_prior = None
        
        # storage parameters
        self.variables_dict = {} #includes feautes and class
        self.features_name = None
        
        # private parameters
        self._initialize = False
        

    def fit(self, dataset):
        self._initialize_fun(dataset)
        self.partial_fit(dataset)

    def partial_fit(self, dataset):
        if not self._initialize:
            self._initialize_fun(dataset)
        else:
            self._check_dataset(dataset, forze_class = True)
        
        '''
        # we transform the dataset to the codes of the categories
        aux_dataset = self._transform_data_to_codes(dataset)
        
        # and update the values
        for idx, i in aux_dataset.iterrows():
            # this iterative version is too slow, it is proposed to do one that counts per column instead of row
            y = i[self.class_index]
            self.features_count[y,self.features_index+np.array(i[self.features_name],dtype=np.int)]+=1
            self.classes_count[y] += 1
        '''
        
        class_categories = self.variables_dict[self.class_index]
        for idi, i in enumerate(self.features_name):
            counters = dataset.groupby([i,self.class_index])[self.class_index].count()
            feature_categories = self.variables_dict[i]
            for idj, j in enumerate(feature_categories):
                for idk, k in enumerate(class_categories):
                    if (j,k) in counters:
                        self.features_count[idk,idj+self.features_index[idi]] += counters[(j,k)]
                        self.classes_count[idk] += counters[(j,k)]
        
        # For the alpha
        self.features_count+=self.alpha

        # Store actual logs probs for efficiency in test
        self.features_probs = np.log(self.features_count/self.classes_count[:, None])
        self.classes_probs = np.log(self.classes_count/sum(self.classes_count))

    def _predict_probs_base(self, dataset):
        self._check_dataset(dataset)
        aux_dataset = self._transform_data_to_codes(dataset)
        return self.classes_probs[:, None] + np.sum(self.features_probs[:,self.features_index+aux_dataset[self.features_name]], axis=2)

    def predict_probs(self, dataset):
        probs = self._predict_probs_base(dataset)
        return np.exp(probs - logsumexp(probs,axis=0))

    def predict_class(self, dataset):        
        aux=self._predict_probs_base(dataset)
        return np.array(range(0, self.class_values))[np.argmax(aux, axis=0)]

    def _initialize_fun(self,dataset):
        
        # Check the class
        if self.class_index is None:
            warnings.warn("WARNING: no class variable selected, look at Class variable.")
            if 'Class' in dataset.columns:
                self.class_index = 'Class'
            else:
                self.class_index = dataset.columns[-1]
                warnings.warn("WARNING: no Class variable, last column used by default.")
                
        if dataset.dtypes[self.class_index] != 'category':
            raise AttributeError( "class must be categorical variable" )
        
        self.class_values=len(dataset[self.class_index].cat.categories)
        self.classes_count=np.zeros((self.class_values,), dtype=np.float)
        self.classes_probs=np.zeros((self.class_values,), dtype=np.float)
        
        self.variables_dict = {}
        self.variables_dict[self.class_index] = dataset[self.class_index].cat.categories
        
        # Check the features
        self.features_name = np.array(dataset.columns)
        self.features_name = np.delete(self.features_name,np.where(self.features_name==self.class_index))
        
        self.features_values=0
        self.features_bins=np.array([],dtype=np.int)
        self.features_index=np.array([0],dtype=np.int)
        
        for idx,i in dataset.dtypes.iteritems():
            if idx != self.class_index:
                if (i=='category'):
                    self.features_bins = np.append(self.features_bins,len(dataset[idx].cat.categories))
                    
                    self.features_values+=len(dataset[idx].cat.categories)
                    self.features_index=np.append(self.features_index,self.features_values)
                    
                    self.variables_dict[idx] = dataset[idx].cat.categories
                    
                else:
                    raise AttributeError( "this implementation of naive_bayes only accepts category variables" )
                    
        self.features_index=self.features_index[0:-1]
        self.features_count=np.zeros((self.class_values,self.features_values), dtype=np.float)
        self.features_probs=np.zeros((self.class_values,self.features_values), dtype=np.float)
        
        self._initialize = True
        

    def _check_dataset(self, dataset, forze_class=False):
        train_set = set(self.variables_dict.keys())
        check_set = set(dataset.columns)
        inter_set = train_set.difference(check_set)
        
        if bool(inter_set) & ( (inter_set != set([self.class_index])) | forze_class ):
            raise AttributeError("structure error: there are variables that don't match with trained model")
        
        for idx,i in dataset.dtypes.iteritems():
            if (i != 'category') | (dataset[idx].cat.categories.tolist() != self.variables_dict[idx].tolist()):
                raise AttributeError("attribute error: {} is not a category variable or its categories are wrong".format(idx))

    def _transform_data_to_codes(self,dataset):
        aux_dataset = dataset.copy()
        for i in aux_dataset.columns:
            aux_dataset[i] = aux_dataset[i].cat.codes
            
        return aux_dataset