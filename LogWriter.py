import os
import csv
import matplotlib.pyplot as plt
from StepEnum import *

def initLog() -> None :
    if (os.path.exists("./log/NormLog.csv")) :
        os.remove("./log/NormLog.csv")

def writeAllNormLog(logList : list) -> None :
    filePath = "./log/NormLog.csv"
    if (os.path.exists(filePath)) :
        os.remove(filePath)

    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as normLog :
            csvWriter = csv.writer(normLog)
            csvWriter.writerow(["K", "Norm"])
    with open(filePath, "+a") as normLog :
        csvWriter = csv.writer(normLog)
        for k in range(0, len(logList)) :
            csvWriter.writerow([k, logList[k]])

def writeClassificationLog(file_name : str, dataset_name : str, resultsList : list) -> None :
    mainDirectoryPath = "./log/" + dataset_name
    if (not os.path.isdir(mainDirectoryPath)) :
        os.mkdir(mainDirectoryPath)
    
    filePath = mainDirectoryPath + "/" + file_name + "_Results.csv"
    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as logFile :
            csvWriter = csv.writer(logFile)
            csvWriter.writerow(["Classified", "Real"])
    
    with open(filePath, "+a") as classificationLog :
        csvWriter = csv.writer(classificationLog)
        for couple in resultsList :
            csvWriter.writerow([str(couple[0]), str(couple[1])])

def writeAccuracyLog(file_name : str, dataset_name : str, accuracy : float, steps_num : int, with_saga : bool, method : StepEnum) :
    mainDirectoryPath = "./log/" + dataset_name
    if (not os.path.isdir(mainDirectoryPath)) :
        os.mkdir(mainDirectoryPath)
    
    filePath = mainDirectoryPath + "/" + file_name + "_Log.csv"
    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as logFile :
            csvWriter = csv.writer(logFile)
            csvWriter.writerow(["Accuracy", "StepsNum", "SAGA", "Method"])
    
    with open(filePath, "+a") as logFile :
            csvWriter = csv.writer(logFile)
            csvWriter.writerow([accuracy, steps_num, with_saga, method.name])

def cartesian_plot(x : list, y : list, x_label : str, y_label : str, title : str) -> None :
    plt.figure(figsize = (50, 9), tight_layout = True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig("./log/plots/" + title)
    plt.clf()
    return

def bar_plot(x : list, y : list, x_label : str, y_label : str, title : str) -> None :
    plt.figure(figsize = (9, 9), tight_layout = True)
    plt.bar(x, y, width = 0.6, color = ['red', 'black'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("./log/plots/" + title)
    plt.clf()
    return

def pie_plot(x: list, y : list, title : str) -> None :
    plt.figure(figsize = (9, 9), tight_layout = True)
    fig = plt.figure(figsize = (7, 7))
    plt.pie(x, labels = y, autopct = '%1.1f%%')
    plt.title(title)
    plt.savefig("./log/plots/" + title)
    plt.clf()
    return