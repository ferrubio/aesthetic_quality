# -*- coding: utf-8 -*-

"""
 The :mod:`Preprocess.Discretize`
 module implements a discretization step for continuous data.
 It discretizes a range of numeric attributes in the dataset
 into nominal attributes. Two unsupervised discretization are
 available: by distance and by frequency.
 
 It is supposed that dataset input parameters are pandas
 DataFrames.
"""

# Author: Fernando Rubio <fernando.rubio@uclm.es>
#

import numpy as np
import pandas as pd
import warnings


class Discretize(object):
    '''
    ================================================
    == Parameters file =============================
    ================================================
    <preprocess>
        <module>Preprocess</module>
        <name>MDL_method</name>
        <params>
            <index>None</index>
            <frequency>False</frequency>
            <bins>3</bins>
        </params>
    </preprocess>

    Parameters
    ----------
    index : array-like, optional (default=None)
        array of indexes, where each one represents a variable in data.
        By default the process is applied to all numeric variables.
    better_encoding : bool, optional (default=False)
        
    class_index : string, optional (default=None)
        name of the column (pandas format).
        By default it look at 'Class' variable. If it doesn't find it, 
        the last column is used. If this last variable is numeric it causes
        a fault.
    '''

    def __init__(self):
        # input parameters
        self.index = None
        
        # storage parameters
        self.cuts_dict = {}


    def train(self, dataset):
        if bool(self.cuts_dict):
            warnings.warn("WARNING: training alredy done, reinitializating.")
            self.cuts_dict = {}
        
        numeric_columns = np.array(dataset.select_dtypes(include=[np.number]).columns)
        if self.index is None:
            self.index = numeric_columns
            if self.index.size == 0:
                warnings.warn("WARNING: no numeric variables for discretization process.")
                return
        else:
            # here I would test the indexes and the names, it depends on the format of the index
            for i in self.index:
                if dataset.domains[i]['type'] != 'numeric':
                    raise AttributeError("Error: selected index are not numeric")
        
        for i in range(0, len(self.index)):
            
            aux_cuts = self.getCuts(dataset,i)
                
            if aux_cuts is None:
                self.cuts_dict[self.index[i]] = [float('-inf'),float('inf')]
            else:
                self.cuts_dict[self.index[i]] = np.append(np.append(float('-inf'),aux_cuts),float('inf'))
                
                
    def process(self, dataset):

        # we will retunr a copy of the dataset with the variables in dict_cuts discretized
        disc_dataset = dataset.copy()
        
        # 
        for i in disc_dataset.columns:
            if i in self.cuts_dict:
                self._apply_disc(i,disc_dataset[i])
                disc_dataset[i] = disc_dataset[i].astype(int)
                disc_dataset[i] = pd.Categorical(disc_dataset[i],range(0,len(self.cuts_dict[i])-1))
                disc_dataset[i] = disc_dataset[i].cat.rename_categories(self._get_categories(i))
            else:
                warnings.warn("WARNING: {} variable is not trained, it will be skipped.".format(i))
                    
        return disc_dataset
        
    def _apply_disc(self,index,data):
        no_changed = np.ones(data.shape[0], dtype=np.bool)
        aux_cuts = self.cuts_dict[index]
        
        for k in range(0, len(aux_cuts)-1):
            to_change = no_changed & (data > aux_cuts[k]) & (data <= aux_cuts[k+1])
            data[to_change] = k
            no_changed = no_changed & ~to_change
            
            
    def _get_categories(self,index):
        aux_cuts = self.cuts_dict[index]
        categories_names = []
        for i in range(0, len(aux_cuts)-2):
            categories_names.append("({:}, {:}]".format(aux_cuts[i],aux_cuts[i+1]))
        categories_names.append("({:}, {:})".format(aux_cuts[len(aux_cuts)-2],aux_cuts[len(aux_cuts)-1]))
                   
        return categories_names
