# -*- coding: utf-8 -*-

"""
 The :mod:`Preprocess.MDL_method`
 module implements a discretization step for continuous data.
 It discretizes a range of numeric attributes in the dataset
 into nominal attributes. Discretization is by Fayyad and
 Irani's MDL method.
 
 It is supposed that dataset input parameters are pandas
 DataFrames.
"""

# Author: Fernando Rubio <fernando.rubio@uclm.es>
#

import numpy as np
import pandas as pd
import warnings

from preprocess.discretize import Discretize

# Is more quicker calculate directly the logs
#max_log = 10000
#log_cache=np.append(0,np.log(range(1,max_log))*range(1,max_log))
log2=np.log(2)

class MDL_method(Discretize):
    '''
    ================================================
    == Parameters file =============================
    ================================================
    <preprocess>
        <module>Preprocess</module>
        <name>MDL_method</name>
        <params>
            <index>None</index>
            <better_encoding>False</better_encoding>
            <class_index>None</class_index>
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
        self.better_encoding = False
        self.class_index = None
        
        # private parameters for reduce memory cost in recursivity
        self._stored_classes = None
        self._stored_data = None


    def train(self, dataset):
        
        if self.class_index is None:
            warnings.warn("WARNING: no class variable selected, look at Class variable.")
            if 'Class' in dataset.columns:
                self.class_index = 'Class'
            else:
                self.class_index = dataset.columns[-1]
                warnings.warn("WARNING: no Class variable, last column used by default.")
                
        if dataset.dtypes[self.class_index] != 'category':
            raise AttributeError( "class must be categorical variable" )
                    
        Discretize.train(self,dataset)
        

    def getCuts(self,dataset,i):
        num_cases, num_columns = dataset.shape
        print(self.index[i])
        data_to_cut = dataset.sort_values([self.index[i]]).loc[:,[self.index[i],self.class_index]]
            
        # this variables are for reduce memory in the recursive calls of cutPointsForSubset
        self._stored_classes = np.array(data_to_cut[self.class_index])
        self._stored_data = np.array(data_to_cut[self.index[i]])

        return self.cutPointsForSubset(0,num_cases)
                
    def cutPointsForSubset(self,first,lastPlusOne):
        if (lastPlusOne-first)<2:
            return None

        numCutPoints = 0
        num_classes = len(np.unique(self._stored_classes))
        num_instances=lastPlusOne-first
        
        # Compute class counts.
        counts = np.zeros((2,num_classes),dtype=np.int)
        for i in range(0,num_classes):
            counts[1,i]=sum(self._stored_classes[first:lastPlusOne]==i)

        # Save prior counts
        priorCounts = counts[1].copy()

        # Entropy of the full set
        priorEntropy = self.entropy(priorCounts)
        bestEntropy = priorEntropy

        #name_index=dataset.data.columns[index]
        # Copy and sort data in index
        #sort_data=np.array(dataset.getColumns([index,dataset.outputIdx]).sort_values(name_index))

        # Find best entropy.
        for i in range(first,lastPlusOne-1):
            counts[0,self._stored_classes[i]] += 1
            counts[1,self._stored_classes[i]] -= 1
            
            # for test speed
            #return counts
        
            if self._stored_data[i] < self._stored_data[i+1]:
                currentCutPoint = (self._stored_data[i] + self._stored_data[i+1]) / 2.0
                currentEntropy = self.entropyConditionedOnRows(counts)

                if currentEntropy < bestEntropy:
                    bestCutPoint = currentCutPoint
                    bestEntropy = currentEntropy
                    bestIndex = i
                    bestCounts=counts.copy()

                numCutPoints+=1

        if not self.better_encoding:
            numCutPoints = (lastPlusOne - first) - 1

        # Checks if gain is zero
        gain = priorEntropy - bestEntropy
        if gain <= 0:
            return None

        # Check if split is to be accepted
        if self.FayyadAndIranisMDL(priorCounts, bestCounts, num_instances, numCutPoints):

            # Select split points for the left and right subsets
            left = self.cutPointsForSubset(first, bestIndex + 1)
            right = self.cutPointsForSubset(bestIndex + 1,lastPlusOne)

            # Merge cutpoints and return them
            if (left is None) & (right is None):
                cutPoints = bestCutPoint

            elif right is None:
                cutPoints = np.append(left,bestCutPoint)

            elif left is None:
                cutPoints = np.append(bestCutPoint,right)

            else:
                cutPoints = np.append(np.append(left, bestCutPoint),right)

            return cutPoints

        else:
            return None


    def entropyConditionedOnRows(self,matrix):

        total=np.sum(matrix)
        if total==0:
            return 0

        global log2
        #vfunc = np.vectorize(self.lnFunc, otypes=[np.float])
        #returnValue= np.sum(np.sum(vfunc(matrix),axis=1) - vfunc(np.sum(matrix,axis=1)))
        
        
        inter_value = np.sum(matrix,axis=1)
        inter_logs = np.zeros(inter_value.shape)
        value_indexes = inter_value > 0
        inter_logs[value_indexes] = np.log(inter_value[value_indexes])*inter_value[value_indexes]
        
        log_matrix = np.zeros(matrix.shape)
        value_indexes = matrix > 0
        log_matrix[value_indexes] = np.log(matrix[value_indexes])*matrix[value_indexes]
                
        returnValue = np.sum(np.sum(log_matrix,axis=1) - inter_logs)
        
        '''
        global log_cache
        
        inter_value = np.sum(matrix,axis=1)
        inter_value[inter_value < 0] = 0
        
        inter_logs = np.zeros(inter_value.shape)
        
        big_indexes = inter_value >= max_log
        inter_logs[big_indexes] = np.log(inter_value[big_indexes])*inter_value[big_indexes]
        inter_logs[~big_indexes] = log_cache[inter_value[~big_indexes]]
        
        
        matrix[matrix < 0] = 0
        
        log_matrix = np.zeros(matrix.shape)
                
        big_indexes = matrix >= max_log
        log_matrix[big_indexes] = np.log(matrix[big_indexes])*matrix[big_indexes]
        log_matrix[~big_indexes] = log_cache[matrix[~big_indexes]]
        
        returnValue = np.sum(np.sum(log_matrix,axis=1) - inter_logs)
        '''
        
        return -returnValue / (total*log2)

    def entropy(self,vector):
        
        vector_sum = np.sum(vector)

        if vector_sum == 0:
            return 0
        
        else:
            global log2        
            
            #vfunc = np.vectorize(self.lnFunc, otypes=[np.float])
            #returnValue=np.sum(-vfunc(vector))
            #vector_sum=np.sum(vector)
            #return (returnValue+self.lnFunc(vector_sum)) / (vector_sum*log2)
        
            log_vector = np.zeros(vector.shape)
            log_vector[vector>0] = np.log(vector[vector>0])*vector[vector>0]

            vector_log = vector_sum * np.log(vector_sum) if vector_sum > 0 else 0
            
            return ( np.sum(-log_vector) + vector_log) / (vector_sum*log2)        
        
    '''    
    def lnFunc(self,value):
        global log_cache
        if value <= 0:
            return 0
        elif value < len(log_cache):
            return log_cache[int(value)]
        else:
            return value*np.log(value)
    '''

    def FayyadAndIranisMDL(self, priorCounts, bestCounts, numInstances, numCutPoints):

        # Compute entropy before split.
        priorEntropy = self.entropy(priorCounts)

        # Compute entropy after split.
        entropy = self.entropyConditionedOnRows(bestCounts)

        # Compute information gain.
        gain = priorEntropy - entropy

        # Number of classes occuring in the set
        numClassesTotal = sum(priorCounts>0)

        # Number of classes occuring in the left subset
        numClassesLeft = sum(bestCounts[0]>0)

        # Number of classes occuring in the right subset
        numClassesRight = sum(bestCounts[1]>0)

        # Entropy of the left and the right subsets
        entropyLeft = self.entropy(bestCounts[0])
        entropyRight = self.entropy(bestCounts[1])

        # Compute terms for MDL formula
        delta = np.log2(np.power(3, numClassesTotal) - 2) - ((numClassesTotal * priorEntropy) -
                                                            (numClassesRight * entropyRight) -
                                                            (numClassesLeft * entropyLeft))

        # Check if split is to be accepted
        return (gain > (np.log2(numCutPoints) + delta) / numInstances)
