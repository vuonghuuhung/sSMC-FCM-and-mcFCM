import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

def data_overiew(df, message):
    print(f'{message}:')
    print('Number of rows: ', df.shape[0])
    print("Number of features:", df.shape[1])
    print("Data Features:")
    print(df.columns.tolist())
    print("Missing values:", df.isnull().sum().values.sum())
    print("Unique values:")
    print(df.nunique())

def initData(linkdata):
    df = pd.read_csv(linkdata, sep=',')
    name_label = df.columns.tolist()[-1]
    labels = pd.DataFrame(df[name_label])

    numClusters = df[name_label].nunique()
    list_label = labels[name_label].unique()
    dict_label = {a:b for a,b in zip(list_label,range( len(list_label) ) )}
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
    return data,numClusters,labels

def calculate_fuzziness(data,numClusters=2,mL=1.5,mU=2):
    mount = len(data)//numClusters
    lamda_point=[]
    for point1 in data:
        dis =[]
        for point2 in data:
            dis.append(np.linalg.norm(point1-point2))
        dis.sort()
        lamda_point.append(np.mean(dis[1:mount]))

    lamda_point = np.array(lamda_point)
    lamda_max = np.max(lamda_point)
    lamda_min = np.min(lamda_point)
    lamda_mid = np.median(lamda_point)
    alpha = np.log(0.5)/np.log((lamda_mid-lamda_min)/(lamda_max-lamda_mid))

    temp = [((a - lamda_min)/(lamda_max-lamda_min))**alpha for a in lamda_point]
    fuzziness = [mL + (mU-mL)*t for t in temp]
    return np.array(fuzziness)

def initCentre(data,numClusters):
    centre = []
    idx = np.random.randint(len(data))
    centre.append(data[idx])
    for i in range(numClusters-1):
        distances = []
        for point in data:
            dis_point_cen = []
            for cen in centre:
                dis_point_cen.append(np.linalg.norm(point-cen))
            distances.append(np.min(dis_point_cen))
        idx = np.argmax(distances)
        centre.append(data[idx])


    return np.array(centre)

def updateU(data,centre,fuzziness):
    degree = []
    i = 0
    for point in data:
        degreePoint = []
        for centroid in centre:
            if np.linalg.norm(point-centroid) == 0.0:
                degreePoint.append(1)
            else:
                sum = 0.0
                p = 2/ (fuzziness[i]-1)
                for cen in centre:
                    if np.linalg.norm(point - cen)!=0.0:
                        temp = np.linalg.norm(point-centroid)/np.linalg.norm(point - cen)
                        temp = temp**p
                        sum +=temp
                    else:
                        sum =np.inf
                        break
                degreePoint.append(1/sum)
        degree.append(degreePoint)
        i+=1
    return np.array(degree)

def calculate_centre(data,centre,degree,fuzziness):
    temp = [[degree[i][j]**fuzziness[i] for j in range(len(centre))] for i in range(len(data))]
    newCentre = []
    for i in range(len(centre)):
        numerator = np.zeros(len(data[0]))
        denominator = 0.0
        for j in range(len(data)):
            numerator += temp[j][i]*data[j]
            denominator += temp[j][i]
        newCentre.append(numerator/denominator)
    diff = np.linalg.norm(newCentre - centre)
    return np.array(newCentre),diff

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

def synchronize_label(labels,clus_label,numClusters,centre):
    max_accur =0.0
    list_ = list(range(numClusters))
    new_labels = clus_label
    newidx = list_
    for temp_ in hoanvi(list_):
        dict_rep = {a:b for a,b in zip(list_,temp_)}
        temp_labels = [dict_rep[a] for a in clus_label]
        accur = metrics.accuracy_score(temp_labels,labels) 
        if accur >max_accur:
            max_accur = accur
            new_labels = temp_labels
            newidx = temp_

    newcentre = centre.copy()
    for i,j in zip(newidx,list_):
        newcentre[i] = centre[j]
    
    return new_labels,newcentre
def mcfcm(dataname='iris',mL=2,mU=4):
    linkdata = ".\data\\"+dataname+".csv"
    data,numClusters,labels = initData(linkdata)
    fuzziness=calculate_fuzziness(data,numClusters,mL,mU)
    centre = initCentre(data,numClusters)
    diff = 100
    epsilon = 0.00005
    while diff > epsilon:
        degree = updateU(data,centre,fuzziness)
        centre,diff = calculate_centre(data, centre, degree, fuzziness)


    clus_label = np.array([np.argmax(degree[i]) for i in range(len(degree))])
    clus_label,centre = synchronize_label(labels,clus_label,numClusters,centre)

    # print(labels)
    # print(clus_label)
    # print(centre)
    # print(metrics.rand_score(clus_label,labels))
    # print(metrics.accuracy_score(clus_label,labels))
    return data,centre,labels,clus_label

if __name__ == '__main__':
    mcfcm('iris')


