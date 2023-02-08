import MetricsCalculate
import mcfcm
import ssmcfcm

def loop_for_mcfcm(dataname,mL,mU):
    getAccur_sum = 0
    getRanIdx_sum = 0
    getDB_sum = 0
    getDunnIndex_sum = 0
    getASWC_sum = 0
    for i in range (10):
        data,centre,labels,clus_label,looptimes = mcfcm.mcfcm(dataname,mL,mU)
        list_metrics = MetricsCalculate.getmetrics(labels,clus_label,centre,data)
        getAccur_sum += list_metrics[0]
        getRanIdx_sum += list_metrics[1]
        getDB_sum += list_metrics[2]
        getDunnIndex_sum += list_metrics[3]
        getASWC_sum += list_metrics[4]
        print("Time " + str(i) + ":\n")
        print("Accuracy: " + str(list_metrics[0]) + "\n")
        print("Rand Index: " + str(list_metrics[1]) + "\n")
        print("Davis-Bouldin: " + str(list_metrics[2]) + "\n")
        print("Dunn) Index: " + str(list_metrics[3]) + "\n")
        print("ASWC: " + str(list_metrics[4]) + "\n")
        print("----------------------------------\n")
    Accur_average = getAccur_sum / 10
    RanIdx_average = getRanIdx_sum / 10
    DB_average = getDB_sum / 10
    DunnIndex_average = getDunnIndex_sum / 10
    ASWC_average = getASWC_sum / 10
    print("Final:\n")
    print("Accuracy: " + str(round(Accur_average, 6)) + "\n")
    print("Rand Index: " + str(round(RanIdx_average, 6)) + "\n")
    print("Davis-Bouldin: " + str(round(DB_average, 6)) + "\n")
    print("Dunn Index: " + str(round(DunnIndex_average, 6)) + "\n")
    print("ASWC: " + str(round(ASWC_average, 6)) + "\n")

def loop_for_ssmcfcm(dataname,mL,mU,percent):
    getAccur_sum = 0
    getRanIdx_sum = 0
    getDB_sum = 0
    getDunnIndex_sum = 0
    getASWC_sum = 0
    for i in range (10):
        data,centre,labels,clus_label,supervise,looptimes = ssmcfcm.ssmcfcm(dataname,mL,mU, percent)
        list_metrics = MetricsCalculate.getmetrics(labels,clus_label,centre,data)
        getAccur_sum += list_metrics[0]
        getRanIdx_sum += list_metrics[1]
        getDB_sum += list_metrics[2]
        getDunnIndex_sum += list_metrics[3]
        getASWC_sum += list_metrics[4]
        print("Time " + str(i) + ":\n")
        print("Accuracy: " + str(list_metrics[0]) + "\n")
        print("Rand Index: " + str(list_metrics[1]) + "\n")
        print("Davis-Bouldin: " + str(list_metrics[2]) + "\n")
        print("Dunn Index: " + str(list_metrics[3]) + "\n")
        print("ASWC: " + str(list_metrics[4]) + "\n")
        print("----------------------------------\n")
    Accur_average = getAccur_sum / 10
    RanIdx_average = getRanIdx_sum / 10
    DB_average = getDB_sum / 10
    DunnIndex_average = getDunnIndex_sum / 10
    ASWC_average = getASWC_sum / 10
    print("Final:\n")
    print("Accuracy: " + str(round(Accur_average, 6)) + "\n")
    print("Rand Index: " + str(round(RanIdx_average, 6)) + "\n")
    print("Davis-Bouldin: " + str(round(DB_average, 6)) + "\n")
    print("Dunn Index: " + str(round(DunnIndex_average, 6)) + "\n")
    print("ASWC: " + str(round(ASWC_average, 6)) + "\n")

loop_for_mcfcm('iris.csv',1.7,2.1)
# loop_for_ssmcfcm('glass.csv',2,7,20)+