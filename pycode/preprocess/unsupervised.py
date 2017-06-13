# -*- coding: utf-8 -*-

"""
 The :mod:`Preprocess.Unsupervised_method`
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

from preprocess.discretize import Discretize

class Unsupervised_method(Discretize):
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
        Discretize.__init__(self)
        
        # Specific input parameters
        self.frequency = False
        self.bins = 3                

    def getCuts(self,dataset,i):
        if self.frequency:
            return self._cut_by_frequency(dataset[self.index[i]])
        else:
            return self._cut_by_distance(dataset[self.index[i]])
                        
    
    def _cut_by_distance(self,data):
        min_value = np.min(data)
        max_value = np.max(data)
        step_value = (max_value - min_value) / self.bins
        aux_cuts = np.arange(min_value, max_value, step_value)
        return aux_cuts[1:]
        
    def _cut_by_frequency(self,data):
        ordered_data = np.sort(data)
        num_cases = ordered_data.shape[0]
        step_value = num_cases / self.bins
        
        aux_cuts = np.zeros(self.bins-1)
        for i in range(0,self.bins-1):
            aux_cuts[i] = ordered_data[int(step_value*(i+1))]
            
        return aux_cuts