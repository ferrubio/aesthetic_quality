import re
import numpy as np
import pandas as pd

def readARFF(path):
    f = open(path, 'rb')
    text = f.readlines()
    f.close()
    data = {"name":None, "vars":[], "domains":[], "output":None, "data":[]}
    categorical = []
    attributeRegex = re.compile("@attribute\s+(\w+)\s+(\w+|\{.*\})")
    for i in range(len(text)):
        l = text[i].strip().decode("utf-8") 
        if(len(l)==0 or l[0]=='%'):
            continue
        if('@relation' in l.lower()):
            data["name"] = l[10:]
        elif('@attribute' in l.lower()):
            matches = attributeRegex.match(l).groups()
            if(matches[1][0]=='{'):#categorical
                categorical.append(True)
                data["vars"].append(matches[0])
                # Remove the {}, split by , and remove spaces (strip)
                domains = {"type":"categorical", "range":list(map(str.strip,matches[1][1:-1].split(",")))}
                data["domains"].append(domains)
            else:
                categorical.append(False)
                data["vars"].append(matches[0])
                domains = {"type":matches[1], "range":[]} #We do not know the ranges, so we read the data later and fix it.
                data["domains"].append(domains)

        elif('@data' in l):
            break

    # First we fix the last attribute as the output
    data["output"] = data["vars"][-1]
    # Transform the lists into numpy arrays
    data["vars"] = np.array(data["vars"])
    data["domains"] = np.array(data["domains"])
    
    # Now the @data part
    for i in range(i+1,len(text)):
        l = text[i].strip().decode("utf-8").split(',')
        if len(l)!=data["vars"].size:
            continue
        l = list(map(str.strip, l))
        data["data"].append(
            tuple([l[i] if not(categorical[i]) else data["domains"][i]["range"].index(l[i]) for i in range(len(l))]))
    dtype = []
    for i in range(data["vars"].shape[0]):
        if data["domains"][i]["type"]=="integer":
            dtype.append((data["vars"][i], np.integer))
        else:
            dtype.append((data["vars"][i], np.float))
    data["data"] = np.array(data["data"], dtype = np.dtype(dtype))
    
    auxDF = pd.DataFrame(data["data"])
    for i in range(0,len(data["vars"])):
        minValue = min(auxDF[data["vars"][i]])
        maxValue = max(auxDF[data["vars"][i]])
        data["domains"][i]["range"]=[minValue,maxValue]
        
    
    # this lines are for mi problem, by default the first variable is the id and I only need the data
    data=pd.DataFrame(data['data'],columns=data['vars'])
    data=data.rename(columns=(lambda x: 'var'+str(int(x[3:])-1)))
    data=data.rename(columns={'var0':'id'})
    
    return data


def votes_probs(votes, prob):

    return np.exp(votes_log_probs(votes,prob))


def votes_log_probs(votes, prob):

    falseVotes=(votes.T*(1-prob))
    trueVotes=(votes.T*prob)

    sumFalseVotes=np.sum(falseVotes,axis=1)
    sumTrueVotes=np.sum(trueVotes,axis=1)

    totalFalseVotes=np.sum(sumFalseVotes)
    totalTrueVotes=np.sum(sumTrueVotes)

    probFalseVotes=np.log(sumFalseVotes.astype(float)/float(totalFalseVotes))
    probTrueVotes=np.log(sumTrueVotes.astype(float)/float(totalTrueVotes))

    probFalseVotes=np.sum(probFalseVotes*votes,axis=1)
    probTrueVotes=np.sum(probTrueVotes*votes,axis=1)

    norm_log=np.logaddexp(probFalseVotes,probTrueVotes)

    return np.array([probFalseVotes-norm_log,probTrueVotes-norm_log])


def getDataset(data):
    final_data=normal_dataset()
    final_data.data=data
    final_data.vars=len(data.columns)
    final_data.domains=np.array([])

    #for i in range(0,final_data.vars):
    #    final_data.domains=np.append(final_data.domains,{'range':[min(data.iloc[:,i]),max(data.iloc[:,i])],'type':'numeric'})

    final_data.domains=\
        np.array([{'range':[min(data.iloc[:,i]),max(data.iloc[:,i])],'type':'numeric'} for i in range(0,final_data.vars)])

    return final_data


def balanced_accuracy(y_true, y_pred):
    y_class = (y_pred >= 0.5).astype(int)
    TP = np.sum((y_true == y_class) & (y_true == 1))
    TN = np.sum((y_true == y_class) & (y_true == 0))
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    return 0.5*(TP/P) + 0.5*(TN/N)


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

