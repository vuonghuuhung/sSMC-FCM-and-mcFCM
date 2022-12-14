from sklearn import metrics

def getAccur(labels,cluslabels):
    return metrics.accuracy_score(labels,cluslabels)

def getRanIdx(labels,cluslabels):
    return metrics.rand_score(labels,cluslabels)

def getmetrics(labels,cluslabels):
    list_metrics = [0]*5
    list_metrics[0] = getAccur(labels,cluslabels)
    list_metrics[1] = getRanIdx(labels,cluslabels)
    return list_metrics
