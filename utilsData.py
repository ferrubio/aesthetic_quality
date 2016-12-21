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
        l = text[i].strip()
        if(len(l)==0 or l[0]=='%'):
            continue
        if("@relation" in l.lower()):
            data["name"] = l[10:]
        elif("@attribute" in l.lower()):
            matches = attributeRegex.match(l).groups()
            if(matches[1][0]=='{'):#categorical
                categorical.append(True)
                data["vars"].append(matches[0])
                # Remove the {}, split by , and remove spaces (strip)
                domains = {"type":"categorical", "range":map(str.strip,matches[1][1:-1].split(","))}
                data["domains"].append(domains)
            else:
                categorical.append(False)
                data["vars"].append(matches[0])
                domains = {"type":matches[1], "range":[]} #We do not know the ranges, so we read the data later and fix it.
                data["domains"].append(domains)

        elif("@data" in l):
            break

    # First we fix the last attribute as the output
    data["output"] = data["vars"][-1]
    # Transform the lists into numpy arrays
    data["vars"] = np.array(data["vars"])
    data["domains"] = np.array(data["domains"])
    # Now the @data part
    for i in range(i+1,len(text)):
        l = text[i].strip().split(',')
        if len(l)!=data["vars"].size:
            continue
        l = map(str.strip, l)
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
    for i in xrange(len(data["vars"])):
        minValue = min(auxDF[data["vars"][i]])
        maxValue = max(auxDF[data["vars"][i]])
        data["domains"][i]["range"]=[minValue,maxValue]
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


