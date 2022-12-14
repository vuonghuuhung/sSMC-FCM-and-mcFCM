import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


def hoanvi(arr):
    if len(arr)==1: return [arr]
    n=len(arr)
    r=[]
    for i in range(n):
        temp = arr[:i]+ arr[i+1:]
        p = hoanvi(temp)
        for y in p:
            r.append([arr[i]]+ y)
    return r

def synchronize_label(labels,clus_label,numClusters):
    max_accur =0.0
    list_ = list(range(numClusters))
    new_labels = clus_label
    for temp_ in hoanvi(list_):
        dict_rep = {a:b for a,b in zip(list_,temp_)}
        temp_labels = [dict_rep[a] for a in clus_label]
        accur = metrics.accuracy_score(temp_labels,labels) 
        if accur >max_accur:
            max_accur = accur
            new_labels = temp_labels
    return new_labels


df = pd.read_csv(".\data\heart.csv", sep=',')
name_label = df.columns.tolist()[-1]
labels = pd.DataFrame(df[name_label])

numClusters = df[name_label].nunique()
list_label = labels[name_label].unique()
dict_label = {a:b for a,b in zip(list_label,range( len(list_label) ) )}
print(list_label,dict_label)
labels = labels.replace(dict_label)
labels = np.array(labels)
labels = labels.reshape((1,len(labels)))
labels = labels[0]

# print(labels)
# print(data)
# data_overiew(df, 'Overview of the dataset')

data = df.drop(name_label,axis = 'columns')
    
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
data = np.array(data)
data = np.array(data)

kmeans=KMeans(n_clusters= numClusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
label_Kmeans = kmeans.fit_predict(data)
print(label_Kmeans)
label_Kmeans = synchronize_label(labels,label_Kmeans,numClusters)
print(metrics.rand_score(label_Kmeans,labels))
print(metrics.accuracy_score(label_Kmeans,labels))