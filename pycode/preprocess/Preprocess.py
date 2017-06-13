import numpy as np

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