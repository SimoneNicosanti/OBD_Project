import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from StepEnum import *

def initLog(dataset_name : str) -> None :
    mainDirectoryPath = "./log/" + dataset_name
    if (not os.path.exists(mainDirectoryPath)) :
        os.mkdir(mainDirectoryPath)
    return mainDirectoryPath


def writeAllNormLog(logList : list, dataset_name : str) -> None :
    mainDirectoryPath = initLog(dataset_name)
    
    filePath = mainDirectoryPath + "/NormLog.csv"
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


def writeErrorLog(errorList : list, dataset_name : str) -> None :
    mainDirectoryPath = initLog(dataset_name)
    
    filePath = mainDirectoryPath + "/ErrorLog.csv"
    if (os.path.exists(filePath)) :
        os.remove(filePath)

    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as normLog :
            csvWriter = csv.writer(normLog)
            csvWriter.writerow(["K", "Error"])
    with open(filePath, "+a") as normLog :
        csvWriter = csv.writer(normLog)
        for k in range(0, len(errorList)) :
            csvWriter.writerow([k, errorList[k]])


def writeClassificationLog(file_name : str, dataset_name : str, resultsList : list) -> None :
    mainDirectoryPath = initLog(dataset_name)
    
    filePath = mainDirectoryPath + "/" + file_name + "_Results.csv"
    if (os.path.exists(filePath)) :
        os.remove(filePath)
        
    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as logFile :
            csvWriter = csv.writer(logFile)
            csvWriter.writerow(["Classified", "Real"])
    
    with open(filePath, "+a") as classificationLog :
        csvWriter = csv.writer(classificationLog)
        for couple in resultsList :
            csvWriter.writerow([str(couple[0]), str(couple[1])])


def writeAccuracyLog(file_name : str, dataset_name : str, accuracy : float, steps_num : int, with_saga : bool, method : StepEnum) :
    mainDirectoryPath = initLog(dataset_name)
    filePath = mainDirectoryPath + "/" + file_name + "_Log.csv"

    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as logFile :
            csvWriter = csv.writer(logFile)
            csvWriter.writerow(["Accuracy", "StepsNum", "SAGA", "Method"])
    
    with open(filePath, "+a") as logFile :
        csvWriter = csv.writer(logFile)
        csvWriter.writerow([accuracy, steps_num, with_saga, method.name])


def residual_plot(residual : pd.DataFrame, dataset_name : str) -> None :
    mainDirectoryPath = initLog(dataset_name)

    figure, axes = plt.subplots(nrows = 1, ncols = 1, tight_layout = True)
    residual_array = residual.values
    # bins_num = int(np.sqrt(len(residual)))
    mean = residual_array.mean()
    std_dev = residual_array.std()
    #bins_num = int(1 + np.log2(len(residual_array)))
    iqr = np.percentile(residual_array, 75) - np.percentile(residual_array, 25)
    bins_num = int((residual_array.max() - residual_array.min()) / (2 * iqr * np.power(len(residual_array), - 1 / 3)))

    axes.hist(residual_array, bins = bins_num, edgecolor = "black", density = True, label = "Residual")

    x_array = np.linspace(residual_array.min(), residual_array.max(), 10000)
    y_array = norm.pdf(x_array, loc = mean, scale = std_dev)
    axes.plot(x_array, y_array, label = "Normal")

    axes.legend()

    figure.savefig(mainDirectoryPath + "/residual_hist")
    #axes.figure.show()
    # res_hist.figure.savefig("./log/" + dataset_name + "/residual_hist")


def cartesian_plot(x : list, y : list, x_label : str, y_label : str, title : str, dataset_name : str) -> None :
    mainDirectoryPath = initLog(dataset_name)
    
    plt.figure(figsize = (50, 9), tight_layout = True)
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.savefig(mainDirectoryPath + "/" + title)
    plt.clf()
    plt.cla()
    return


def bar_plot(x : list, y : list, x_label : str, y_label : str, title : str, dataset_name : str) -> None :
    mainDirectoryPath = initLog(dataset_name)

    plt.figure(figsize = (9, 9), tight_layout = True)
    plt.bar(x, y, width = 0.6, color = ['red', 'black'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(mainDirectoryPath + "/" + title)
    plt.clf()
    plt.cla()
    return


def pie_plot(x: list, y : list, title : str, dataset_name : str) -> None :
    mainDirectoryPath = initLog(dataset_name)
        
    plt.figure(figsize = (9, 9), tight_layout = True)
    fig = plt.figure(figsize = (7, 7))
    plt.pie(x, labels = y, autopct = '%1.1f%%')
    plt.title(title)
    plt.savefig(mainDirectoryPath + "/" + title)
    plt.clf()
    plt.cla()
    return