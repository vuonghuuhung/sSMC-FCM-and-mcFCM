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



def initData(datalink):
    df = pd.read_csv(datalink, sep=',')
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
    return data,numClusters,labels


def supervise_rand(nums,percent=20):
    list_ = []
    for i in range(nums*percent//100):
        temp = np.random.randint(nums)
        while temp in list_:
            temp = np.random.randint(nums)
        list_.append(temp)

    supervise = np.zeros(nums)
    supervise[list_]=1
    return supervise


def initCentre(data,numClusters,supervise,labels):
    centre = np.zeros((numClusters,len(data[0])))
    count = [0]* numClusters
    for i in range(len(data)):
        if supervise[i] ==1:
            centre[labels[i]] += data[i]
            count[labels[i]] += 1 
    for i in range(numClusters):
        if count[i] != 0:
            centre[i] = centre[i]/count[i]     
        else: centre[i] = data[i] 
    return centre


def left(uik,sum_u,mL,mU):
    temp_exp = (mU-mL)/(mU-1)
    denom = (uik+sum_u)**temp_exp
    return uik/denom


def updateU(data,centre,mL,mU,supervise,labels):
    degree = []
    i = 0
    for point in data:
        degreePoint = []
        for centroid in centre:
            if np.linalg.norm(point-centroid) == 0.0:
                degreePoint.append(1)
            else:
                sum = 0.0
                p = 2/ (mL-1)            #Update tất cả U theo công thức không giám sát 
                for cen in centre:       #rồi tính lại theo có giám sát sau
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

    #Tính U có giám sát
    sup = np.where(supervise==1)[0]
    
    for i in sup:
        disPoint_Cen = [np.linalg.norm(data[i]-c) for c in centre]
        dmin = np.min(disPoint_Cen)
        if dmin==0.0:
            u = [0]*len(centre)
            u[np.argmin(disPoint_Cen)] =1
            degree[i] = u
        else:
            disPoint_Cen = disPoint_Cen/dmin
            ui = degree[i]
            for j in range(len(centre)):
                temp_base = 1/(mL*(disPoint_Cen[j]**2))
                temp_exp = 1/(mL-1)
                ui[j] = temp_base**temp_exp

            #Giải uik
            k = labels[i]
            sum_u = np.sum(ui) - ui[k]
            uik=0.0

            temp_base = 1/(mU * (disPoint_Cen[k]**2))
            temp_exp = 1/(mU-1)
            right = temp_base**temp_exp

            while left(uik,sum_u,mL,mU) < right:
                uik+=1
            
            low, high = uik-1,uik
            uik = (low+high)/2
    
            while abs(left(uik,sum_u,mL,mU) -right) > 0.005:
                if left(uik,sum_u,mL,mU) > right:
                    high = uik
                else: low = uik
                uik = (low+high)/2

            
                
            #Chuẩn hoá
            ui[k] = uik
            sum_u = sum_u+uik
            ui = ui/sum_u
            degree[i] = ui

    return np.array(degree)



def calculate_centre(data,centre,degree,mL,mU,supervise,labels):

    temp = []
    for i in range(len(data)):
        temp_point= []
        for j in range(len(centre)):
            if supervise[i] ==1 and labels[i] == j:
                temp_point.append(degree[i][j]**mU)
            else: temp_point.append(degree[i][j]**mL)
        temp.append(temp_point)

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


def ssmcfcm(dataname,mL,mU,percent):

    linkdata = ".\data\\"+dataname
    data,numClusters,labels = initData(linkdata)
    
    supervise =supervise_rand(len(data),percent)
    centre = initCentre(data,numClusters,supervise,labels)
    diff = 100
    epsilon = 0.005
    while diff > epsilon:
        degree = updateU(data,centre,mL,mU,supervise,labels)
        centre,diff = calculate_centre(data, centre, degree,mL, mU,supervise,labels)



    clus_label = [np.argmax(degree[i]) for i in range(len(degree))]

    # print(labels)
    # print(supervise)
    # print(np.array(clus_label))
    # print(centre)
    # print(metrics.rand_score(clus_label,labels))
    # print(metrics.accuracy_score(clus_label,labels))
    return data,centre,labels,clus_label,supervise

if __name__ == '__main__':
    ssmcfcm('glass.csv',2,5,20)


