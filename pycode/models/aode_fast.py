import numpy as np
import pandas as pd
from scipy.misc import logsumexp
import warnings

class AODE_fast():

    '''
    ================================================
    == Parameters file =============================
    ================================================
    <model>
        <module>bayes</module>
        <name>AODE</name>
        <params>
            <get_probs>False</get_probs>
            <alpha>1.0</alpha>
            <fit_prior>True</fit_prior>
            <class_prior>None</class_prior>
            <freq_limit>1.0</freq_limit>
        </params>
    </model>

    Parameters
    ----------
    get_probs: boolean
        if false, return predictions
        else, return a vector of probabilities for classes
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    freq_limit : float, optional (default=1.0)
        Impose a frequency limit for superParents
    '''

    def __init__(self):
        self.class_index = None
        self.alpha = 1
        self.fit_prior = True
        self.class_prior = None
        self.freq_limit=1.0
        self.initialize=False
        self.get_probs=False

    
    def fit(self, dataset):
        self._initialize_fun(dataset)
        self.partial_fit(dataset)

    def partial_fit(self, dataset):

        if not self.initialize:
            self._initialize_fun(dataset)
        else:
            self._check_dataset(dataset)

        class_categories = self.variables_dict[self.class_index]
        counters = dataset.groupby([self.class_index])[self.class_index].count()
        self.classes_count = np.array(counters)
        
        for idi, i in enumerate(self.features_name):
            for idj, j in enumerate(self.features_name):
                counters = dataset.groupby([i,j,self.class_index])[self.class_index].count()
                feature_categories_i = self.variables_dict[i]
                feature_categories_j = self.variables_dict[j]
                
                for idx, x in enumerate(feature_categories_i):
                    for idy, y in enumerate(feature_categories_j):
                        for idk, k in enumerate(class_categories):
                            if (x,y,k) in counters:
                                self.features_count[idk,self.features_index2[idi]+self.features_values*idx
                                                    +self.features_index1[idj]+idy] += counters[(x,y,k)]
                                #self.classes_count[idk] += counters[(x,y,k)]
                                
            # For the frequencies
            counters = dataset.groupby([i])[self.class_index].count()
            feature_categories = self.variables_dict[i]
            for idx, x in enumerate(feature_categories):
                if x in counters:
                    self.frequencies[idx+self.features_index1[idi]] += counters.loc[x]
                        
        

    def _predict_probs_base(self, dataset):

        probs=np.zeros((dataset.shape[0],self.class_values),dtype=np.float)
        counter=0
        
        aux_dataset = self._transform_data_to_codes(dataset)
        
        
        # Non iterative version
        attIndex = np.array(aux_dataset[self.features_name]) + self.features_index1
        for i in range(0,self.class_values):
            init_spodeP = False
            boolean_vector_aux=np.ones(len(self.features_index1), dtype=bool)
            for j in range(0,len(self.features_index1)):
                parents=attIndex[:,j]
                
                #if(self.frequencies[parent]<self.freq_limit):
                #    continue

                values_start_aux=self.features_values*parents
                countsForParents=np.array([self.features_count[i][x:x+self.features_values] for x in values_start_aux])
                boolean_vector_aux[j]=False

                classparentfreq = np.array([countsForParents[idx, x] for idx, x in enumerate(parents)])
                spodeP = np.log((classparentfreq + 1.0)/(np.sum(self.classes_count) + self.class_values * \
                                                         len(self.variables_dict[self.features_name[j]])))
                
                spodeP += np.sum(np.log(( np.array([countsForParents[idx,x] for idx, x in enumerate(attIndex[:,boolean_vector_aux])])
                                         +1.0)/
                                        (np.array([self.features_value_vector[boolean_vector_aux]+x for x in classparentfreq]))),
                                 axis=1)
                
                if not(init_spodeP):
                    probs[:, i] = spodeP
                    init_spodeP = True
                else:
                    probs[:, i] = np.logaddexp(spodeP, probs[:, i])

                boolean_vector_aux[j] = True
        
        return np.array(probs,dtype=np.float)

    def predict_probs(self, dataset):
        probs = self._predict_probs_base(dataset)

        #return (probs.T / np.sum(probs, axis=1))

        return np.exp(probs.T - logsumexp(probs,axis=1)).T

    def predict_class(self, dataset):
        aux=self._predict_probs_base(dataset)
        return pd.Categorical(self.variables_dict[self.class_index][np.argmax(aux, axis=1)],self.variables_dict[self.class_index])

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
        self.features_value_vector=np.array([],dtype=np.int)
        self.features_index1=np.array([0],dtype=np.int)
        
        for idx,i in dataset.dtypes.iteritems():
            if idx != self.class_index:
                if (i=='category'):
                    self.features_values+=len(dataset[idx].cat.categories)
                    self.features_value_vector = np.append(self.features_value_vector,len(dataset[idx].cat.categories))
                    self.features_index1=np.append(self.features_index1,self.features_values)
                    
                    self.variables_dict[idx] = dataset[idx].cat.categories
                    
                else:
                    raise AttributeError( "this implementation of AODE only accepts category variables" )
        
        self.features_index1=self.features_index1[0:-1]
        self.features_index2=self.features_index1*self.features_values

        self.features_count=np.zeros((self.class_values,self.features_values*self.features_values), dtype=np.float)
        self.frequencies=np.zeros(self.features_values, dtype=np.float)

        self.initialize=True
        

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