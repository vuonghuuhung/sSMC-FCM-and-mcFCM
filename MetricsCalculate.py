from sklearn import metrics
import numpy as np
def getAccur(labels,cluslabels):
    return metrics.accuracy_score(labels,cluslabels)

def getRanIdx(labels,cluslabels):
    return metrics.rand_score(labels,cluslabels)

def getInnerDistance(cluslabels,centre,data):
    innerDis = np.full(len(centre),0.0)
    for i in range(len(data)):
        cen = cluslabels[i]
        innerDis[cen] += np.linalg.norm(data[i]-centre[cen])

    for i in range(len(centre)):
        innerDis[i] = innerDis[i]/(cluslabels.count(i))
    return innerDis

def getDB(cluslabels,centre,data):
    innerDis = getInnerDistance(cluslabels,centre,data)
    max = 0
    for i in range(len(centre)):
        for j in range(i+1,len(centre)):
        
            score = (innerDis[i]+innerDis[j])/np.linalg.norm(centre[i]-centre[j])
            if max < score:
                max = score
    return max

def getDunnIndex(cluslabels,centre,data):
    inter = np.inf
    for i in range(len(centre)):
        for j in range(i+1,len(centre)):
            temp = np.linalg.norm(centre[i]-centre[j])
            if inter > temp:
                inter = temp

    intra = 0
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            if cluslabels[i] == cluslabels[j]:
                temp = np.linalg.norm(data[i]-data[j])
                if intra < temp:
                    intra = temp
    dunnIdx = inter/intra
    return dunnIdx

def getASWC(cluslabels,centre,data):
    b = []
    a = []
    for i in range(len(data)):
        ai,bi,counta,countb= 0,0,0,0
        for j in range(len(data)):
            if i!=j:
                if cluslabels[i]!=cluslabels[j]:
                    bi +=np.linalg.norm(data[i]-data[j])
                    countb +=1
                else:
                    ai += np.linalg.norm(data[i]-data[j])
                    counta +=1
        b.append(bi/countb)
        a.append(ai/counta)

    temp =0
    for i in range(len(data)):
        temp += b[i]/(a[i]+0.0001)

    return temp/len(data)




def getmetrics(labels,cluslabels,centre,data):
    list_metrics = [0]*5
    list_metrics[0] = getAccur(labels,cluslabels)
    list_metrics[1] = getRanIdx(labels,cluslabels)
    list_metrics[2] = getDB(cluslabels,centre,data)
    list_metrics[3] = getDunnIndex(cluslabels,centre,data)
    list_metrics[4] = getASWC(cluslabels,centre,data)
    return list_metrics
