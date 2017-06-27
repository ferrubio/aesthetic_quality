import numpy as np

def balance_class(classes):
    
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

    #final_indexes = np.sort(final_indexes)

    return np.sort(final_indexes)


def reference_forward_implementation(features, size, k, alpha, beta):
    """A reference implementation of the local response normalization."""
    num_data = features.shape[0]
    channels = features.shape[1]
    output = np.zeros_like(features)
    scale = np.zeros_like(features)
    for n in range(num_data):
        for c in range(channels):
            local_start = c - (size - 1) / 2
            local_end = local_start + size
            local_start = max(local_start, 0)
            local_end = min(local_end, channels)
            scale[n, c] = k + (features[n, int(local_start):int(local_end)]**2).sum() * alpha / size
            output[n, c] = features[n, c] / (scale[n, c] ** beta)
    return output, scale