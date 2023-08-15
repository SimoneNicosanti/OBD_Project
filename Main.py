import pandas as pd
from Utils import *
from LogWriter import *
from DatasetInfo import dataset_dict
from StepEnum import *
from CrossValidator import *

# TODO : tipizzare tutte le variabili ed i return delle funzioni

def main() :
    np.random.seed(123456)

    dataset_name = "Students"

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
    
    layerNumArray : list = [2]
    neuronNumArray : list = [32, 64]
    lambdaL1Array : list = []
    lambdaL2Array : list = [1e-3, 1e-2, 1e-1, 0.0, 1e0, 1e1]
    lambdaL1 = 0.0
    lambdaL2 = 1e-3
    crossValidation = False
    method = StepEnum.NADAM
    epochs = 1000
    with_SAGA = False
    show_error = True
    model, gradient_norm_array, error_array = crossValidate(
        isClassification, 
        layerNumArray, 
        neuronNumArray, 
        lambdaL1Array,
        lambdaL2Array,
        X_train, Y_train, 
        X_valid, Y_valid,
        epochs,
        with_SAGA,
        method,
        lambdaL1,
        lambdaL2,
        show_error,
        crossValidation
    )

    writeAllNormLog(gradient_norm_array)
    writeErrorLog(error_array)

    accuracy_trainining, trainingPredictionArray = model.predict(X_train, Y_train)
    accuracy_generalization, generalizationPredictionArray = model.predict(X_test, Y_test)
    writeClassificationLog("Training", dataset_name, trainingPredictionArray)
    writeClassificationLog("Generalization", dataset_name, generalizationPredictionArray)
    writeAccuracyLog("Training", dataset_name, accuracy_trainining, epochs, with_SAGA, method)
    writeAccuracyLog("Generalization", dataset_name, accuracy_generalization, epochs, with_SAGA, method)

    print("Training Accuracy: ", accuracy_trainining)
    print("Generalization Accuracy:", accuracy_generalization)

    normDataFrame = pd.read_csv("./log/" + dataset_name + "NormLog.csv")
    cartesian_plot(normDataFrame["K"], normDataFrame["Norm"], "Numero di iterazioni", "Norma del gradiente", "Norma del gradiente in funzione del numero di iterazioni")

    errorDataFrame = pd.read_csv("./log/" + dataset_name + "ErrorLog.csv")
    cartesian_plot(errorDataFrame["K"], errorDataFrame["Error"], "Iterazione", "Errore", "ERRORE NEL NUMERO DI ITERAZIONI")

    bar_plot(["Training Accuracy", "Generalization Accuracy"], [accuracy_trainining, accuracy_generalization], "Type of accuracy", "Accuracy", "Bar plot for accuracies")
    pie_plot([len(X_train), len(X_valid), len(X_test)], ["Training Set", "Validation Set", "Test Set"], "Ripartizione dataset")

    if (not isClassification) :
        regression_results = pd.read_csv("./log/" + dataset_name + "/Generalization_Results.csv")
        residual = regression_results["Real"] - regression_results["Classified"]
        residual_plot(residual, dataset_name)

if __name__ == "__main__" :
    main()


