import pandas as pd
from Utils import *
from LogWriter import *
from NeuralNetwork import NeuralNetwork
from DatasetInfo import dataset_dict
from StepEnum import *
from CrossValidator import *

def main() :
    np.random.seed(123456)

    dataset_name = "Cancer"

    dataset_info = dataset_dict[dataset_name]
    dataset = pd.read_csv(dataset_info["fileName"])
    targetName = dataset_info["targetName"]
    toDrop = dataset_info["toDrop"]
    toOHE = dataset_info["toOHE"]
    isClassification = dataset_info["classification"]

    preprocess_function = dataset_info.get("preprocess_function")
    if (preprocess_function != None) :
        preprocess_function(dataset)

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = dataset, targetName = targetName, targetDrop = toDrop, targetOHE = toOHE)
    
    layerNumArray : list = [2, 3]
    neuronNumArray : list = [64, 128, 256]
    crossValidation = False
    method = StepEnum.NADAM
    max_steps = 1000
    with_SAGA = False
    model = crossValidate(
        isClassification, 
        layerNumArray, 
        neuronNumArray, 
        X_train, 
        Y_train, 
        X_valid, 
        Y_valid,
        max_steps,
        with_SAGA,
        method,
        crossValidation = crossValidation
    )

    #print(model.train_y_mean)
    #model.fit(X_train, Y_train, max_steps = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA)

    accuracy_trainining, trainingPredictionArray = model.predict_2(X_train, Y_train)
    accuracy_generalization, generalizationPredictionArray = model.predict_2(X_test, Y_test)
    writeClassificationLog("Training", dataset_name, trainingPredictionArray)
    writeClassificationLog("Generalization", dataset_name, generalizationPredictionArray)
    writeAccuracyLog("Training", dataset_name, accuracy_trainining, max_steps, with_SAGA, method)
    writeAccuracyLog("Generalization", dataset_name, accuracy_generalization, max_steps, with_SAGA, method)

    print("Training Accuracy: ", accuracy_trainining)
    print("Generalization Accuracy:", accuracy_generalization)

    # TODO : istogramma dei residui per problemi di regressione

    normDataFrame = pd.read_csv("./log/NormLog.csv")
    cartesian_plot(normDataFrame["K"], normDataFrame["Norm"], "Numero di iterazioni", "Norma del gradiente", "Norma del gradiente in funzione del numero di iterazioni")

    errorDataFrame = pd.read_csv("./log/ErrorLog.csv")
    cartesian_plot(errorDataFrame["K"], errorDataFrame["Error"], "Iterazione", "Errore", "ERRORE NEL NUMERO DI ITERAZIONI")

    bar_plot(["Training Accuracy", "Generalization Accuracy"], [accuracy_trainining, accuracy_generalization], "Type of accuracy", "Accuracy", "Bar plot for accuracies")
    pie_plot([len(X_train), len(X_valid), len(X_test)], ["Training Set", "Validation Set", "Test Set"], "Ripartizione dataset")

if __name__ == "__main__" :
    main()


