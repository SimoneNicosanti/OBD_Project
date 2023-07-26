import os
import csv
import matplotlib.pyplot as plt

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


def writeClassificationLog(fileName : str, classificationList : list) -> None :
    filePath = "./log/" + fileName + ".csv"

    if (os.path.exists(filePath)) :
        os.remove(filePath)

    if (not os.path.exists(filePath)) :
        with open(filePath, "+x") as classificationLog :
            csvWriter = csv.writer(classificationLog)
            csvWriter.writerow(["Classified", "Real"])

    with open(filePath, "+a") as classificationLog :
        csvWriter = csv.writer(classificationLog)
        for couple in classificationList :
            csvWriter.writerow([str(couple[0]), str(couple[1])])


def cartesian_plot(x : list, y : list, x_label : str, y_label : str, title : str) -> None :
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("./log/" + title)
    plt.clf()
    return


def bar_plot(x : list, y : list, x_label : str, y_label : str, title : str) -> None :
    plt.bar(x, y, width = 0.6, color = ['red', 'black'])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("./log/" + title)
    plt.clf()
    return


def pie_plot(x: list, y : list, title : str) -> None :
    fig = plt.figure(figsize = (7, 7))
    plt.pie(x, labels = y, autopct = '%1.1f%%')
    plt.title(title)
    plt.savefig("./log/" + title)
    plt.clf()
    return