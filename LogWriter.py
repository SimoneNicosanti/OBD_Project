import os
import csv
import matplotlib.pyplot as plt

def initLog() -> None :
    if (os.path.exists("./log/NormLog.csv")) :
        os.remove("./log/NormLog.csv")
    

def writeNormLog(k : int, norm : float) -> None :
    if (not os.path.exists("./log/NormLog.csv")) :
        with open("./log/NormLog.csv", "+x") as normLog :
            csvWriter = csv.writer(normLog)
            csvWriter.writerow(["K", "Norm"])
    else :
        with open("./log/NormLog.csv", "+a") as normLog :
            csvWriter = csv.writer(normLog)
            csvWriter.writerow([k, norm])


def writeAllNormLog(logList : list) -> None :
    if (not os.path.exists("./log/NormLog.csv")) :
        with open("./log/NormLog.csv", "+x") as normLog :
            csvWriter = csv.writer(normLog)
            csvWriter.writerow(["K", "Norm"])
    with open("./log/NormLog.csv", "+a") as normLog :
        csvWriter = csv.writer(normLog)
        for k in range(0, len(logList)) :
            csvWriter.writerow([k, logList[k]])


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