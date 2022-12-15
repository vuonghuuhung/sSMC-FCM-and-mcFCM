import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# Overview data: show current status of data after read from file 
def data_overiew(df, message):
    print(f'{message}:')
    print('Number of rows: ', df.shape[0])
    print("Number of features:", df.shape[1])
    print("Data Features:")
    print(df.columns.tolist())
    print("Missing values:", df.isnull().sum().values.sum())
    print("Unique values:")
    print(df.nunique())

# Initialize data to clean the data 
def initData(linkdata):
    df = pd.read_csv(linkdata, sep=',') # read file with link of data and separate by comma

    name_label = df.columns.tolist()[-1] # take the name of labels column which is from the last column
    labels = pd.DataFrame(df[name_label]) # from the name of column name_label, take the value of that column

    numClusters = df[name_label].nunique() # count unique labels and assgin to equal num of clusters
    list_label = labels[name_label].unique() # take all the unique labels name 
    dict_label = {a:b for a,b in zip(list_label,range( len(list_label) ) )} # assign id to labels
    labels = labels.replace(dict_label) # replace labels from string to id 
    # take an array of labels id
    labels = np.array(labels) 
    labels = labels.reshape((1,len(labels)))
    labels = labels[0]

    # data_overiew(df, 'Overview of the dataset')

    data = df.drop(name_label,axis = 'columns') # delete the label column

    # scale data to fit with all column     
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)   

    return data,numClusters,labels

# Function calculate the fuzziness number 
def calculate_fuzziness(data,numClusters=2,mL=1.5,mU=2):
    mount = len(data)//numClusters # N/C: take N/C data points 
    
    lamda_point=[] # array of distance from a point to N/C nearest points 
    for point1 in data:
        dis =[] # array of distance from a point to other points
        for point2 in data:
            dis.append(np.linalg.norm(point1-point2)) # calculate distance
        dis.sort() # sort array
        lamda_point.append(np.mean(dis[1:mount])) # take from dis the 1st to mount point, pass the dis[0] = its self

    lamda_point = np.array(lamda_point) # refactor array 
    lamda_max = np.max(lamda_point) # get the max lambda
    lamda_min = np.min(lamda_point) # get the min lambda
    lamda_mid = np.median(lamda_point) # get the mid lambda
    alpha = np.log(0.5)/np.log((lamda_mid-lamda_min)/(lamda_max-lamda_mid)) # get alpha

    # calculate fuzziness number for each data point
    temp = [((a - lamda_min)/(lamda_max-lamda_min))**alpha for a in lamda_point] 
    fuzziness = [mL + (mU-mL)*t for t in temp]

    return np.array(fuzziness)

# Function initialize centre randomly
def initCentre(data,numClusters):
    centre = [] # centre arrays
    idx = np.random.randint(len(data))
    centre.append(data[idx]) # randomly append a point to be centre, first centre 
    for i in range(numClusters-1): # append other centres base on the first centre
        distances = [] 
        for point in data: # for each point 
            dis_point_cen = []
            for cen in centre: # for each centre
                dis_point_cen.append(np.linalg.norm(point-cen)) # calculate distance from point to all already appended centre
            distances.append(np.min(dis_point_cen)) # take the minimun of all distance from all current centre to a point
        # take the farthest point among current centre to be the next centre
        idx = np.argmax(distances) 
        centre.append(data[idx])

    return np.array(centre)

# Function update U value, represents for which cluster the data point belongs to
# Follow the form in report
def updateU(data,centre,fuzziness):
    degree = [] # 
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

# Function reposition current centre after update U value 
# Follow the form in report
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
def mcfcm(dataname='iris.csv',mL=2,mU=4):
    # init data
    linkdata = ".\data\\"+dataname
    data,numClusters,labels = initData(linkdata)

    # calculate fuzziness parameter
    fuzziness=calculate_fuzziness(data,numClusters,mL,mU)

    # initialize centre randomly
    centre = initCentre(data,numClusters)

    # iterator 
    diff = 100
    epsilon = 0.00005
    while diff > epsilon:
        degree = updateU(data,centre,fuzziness)
        centre,diff = calculate_centre(data, centre, degree, fuzziness)

    # synchronize label
    clus_label = np.array([np.argmax(degree[i]) for i in range(len(degree))])
    clus_label,centre = synchronize_label(labels,clus_label,numClusters,centre)

    # print(labels)
    # print(np.array(clus_label))
    # print(centre)
    # print(metrics.rand_score(clus_label,labels))
    # print(metrics.accuracy_score(clus_label,labels))
    return data,centre,labels,clus_label

if __name__ == '__main__':
    mcfcm('iris.csv')


