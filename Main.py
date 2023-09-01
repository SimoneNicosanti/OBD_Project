import pandas as pd
from Utils import *
from LogWriter import *
from DatasetInfo import dataset_dict
from StepEnum import *
from CrossValidator import *
import datetime
from os import environ


# TODO : tipizzare tutte le variabili ed i return delle funzioni

def main() :
    np.random.seed(123456)

    dataset_name = "Chinese"

    dataset_info = dataset_dict[dataset_name]
    if (dataset_name == "Chinese") :
        dataset = pd.read_csv(dataset_info["fileName"], compression = "gzip")
    else :
        dataset = pd.read_csv(dataset_info["fileName"])
    targetName = dataset_info["targetName"]
    toDrop = dataset_info["toDrop"]
    toOHE = dataset_info["toOHE"]
    isClassification = dataset_info["classification"]

    preprocess_function = dataset_info.get("preprocess_function")
    if (preprocess_function != None) :
        preprocess_function(dataset)

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetPreprocess(
        dataset = dataset, 
        targetName = targetName, 
        targetDrop = toDrop, 
        targetOHE = toOHE, 
        testSetSize = 0.15, 
        validationSetSize = 0.15
    )
    
    layerNumArray : list = [2]
    neuronNumArray : list = [32, 64]
    lambdaL1Array : list = []
    lambdaL2Array : list = [1e-3, 1e-2, 1e-1, 0.0, 1e0, 1e1]
    lambdaL1 = 0.0
    lambdaL2 = 1e-3
    crossValidation = False
    method = StepEnum.NADAM
    epochs = 5
    with_SAGA = False
    show_error = False
    with_replacement = False
    start = time.time()
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
        crossValidation,
        with_replacement
    )
    end = time.time()

    totalTime = str(datetime.timedelta(seconds = end - start))

    writeAllNormLog(gradient_norm_array, dataset_name)
    if (show_error) :
        writeErrorLog(error_array, dataset_name)

    accuracy_trainining, trainingPredictionArray = model.predict(X_train, Y_train)
    accuracy_generalization, generalizationPredictionArray = model.predict(X_test, Y_test)
    writeClassificationLog("Training", dataset_name, trainingPredictionArray)
    writeClassificationLog("Generalization", dataset_name, generalizationPredictionArray)
    writeAccuracyLog("Training", dataset_name, accuracy_trainining, epochs, with_SAGA, method, totalTime, model.neuronNumArray, with_replacement)
    writeAccuracyLog("Generalization", dataset_name, accuracy_generalization, epochs, with_SAGA, method, totalTime, model.neuronNumArray, with_replacement)

    print("Training Accuracy:", accuracy_trainining)
    print("Generalization Accuracy:", accuracy_generalization)
    print("Training Time:", totalTime)

    normDataFrame = pd.read_csv("./log/" + dataset_name + "/NormLog.csv")
    cartesian_plot(
        normDataFrame["K"], 
        normDataFrame["Norm"], 
        "Numero di iterazioni", 
        "Norma del gradiente", 
        "GRADIENTE NEL NUMERO DI ITERAZIONI",
        dataset_name)

    if (show_error) :
        errorDataFrame = pd.read_csv("./log/" + dataset_name + "/ErrorLog.csv")
        cartesian_plot(
            errorDataFrame["K"], 
            errorDataFrame["Error"], 
            "Iterazione", 
            "Errore", 
            "ERRORE NEL NUMERO DI ITERAZIONI",
            dataset_name)

    #bar_plot(["Training Accuracy", "Generalization Accuracy"], [accuracy_trainining, accuracy_generalization], "Type of accuracy", "Accuracy", "Bar plot for accuracies")
    #pie_plot([len(X_train), len(X_valid), len(X_test)], ["Training Set", "Validation Set", "Test Set"], "Ripartizione dataset")

    if (not isClassification) :
        regression_results = pd.read_csv("./log/" + dataset_name + "/Generalization_Results.csv")
        residual = regression_results["Real"] - regression_results["Classified"]
        residual_plot(residual, dataset_name)

if __name__ == "__main__" :
    environ["OMP_NUM_THREADS"] = "4"
    environ["OPENBLAS_NUM_THREADS"] = "4"
    main()


