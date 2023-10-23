import pandas as pd
from Utils import *
from LogWriter import *
from DatasetInfo import dataset_dict
from StepEnum import *
from CrossValidator import *
import datetime


def main() :
    np.random.seed(123456)

    ## Cambiare qui la stringa con il nome relativo al dataset da utilizzare
    ## (Nomi presenti in DatasetInfo.py)
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
        testSetSize = 0.2, 
        validationSetSize = 0.2
    )
    
    crossLayerNum : list = [2, 3]
    crossNeuronNum : list = [128, 256]
    crossLambdaL1 : list = [0.0]
    crossLambdaL2 : list = [1e-3, 1e-2, 1e-1]
    lambdaL1 = 0.0
    lambdaL2 = 1e-3
    crossValidation = False
    threadValidation = False
    method = StepEnum.NADAM
    epochs = 5
    with_SAGA = False
    show_error = True
    with_replacement = False
    
    model, gradient_norm_array, error_array, training_time = buildModel(
        isClassification, 
        crossLayerNum, 
        crossNeuronNum, 
        crossLambdaL1,
        crossLambdaL2,
        X_train, Y_train, 
        X_valid, Y_valid,
        epochs,
        with_SAGA,
        method,
        lambdaL1,
        lambdaL2,
        show_error,
        crossValidation,
        threadValidation,
        with_replacement
    )

    totalTime = str(datetime.timedelta(seconds = training_time))

    writeAllNormLog(gradient_norm_array, dataset_name)
    if (show_error) :
        writeErrorLog(error_array, dataset_name)

    accuracy_trainining, trainingPredictionArray = model.test(X_train, Y_train)
    accuracy_generalization, generalizationPredictionArray = model.test(X_test, Y_test)
    writeClassificationLog("Training", dataset_name, trainingPredictionArray)
    writeClassificationLog("Generalization", dataset_name, generalizationPredictionArray)
    writeAccuracyLog("Training", dataset_name, accuracy_trainining, epochs, with_SAGA, method, totalTime, model.neuronNumArray, with_replacement, model.lambdaL1, model.lambdaL2)
    writeAccuracyLog("Generalization", dataset_name, accuracy_generalization, epochs, with_SAGA, method, totalTime, model.neuronNumArray, with_replacement, model.lambdaL1, model.lambdaL2)

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

    if (not isClassification) :
        regression_results = pd.read_csv("./log/" + dataset_name + "/Generalization_Results.csv")
        residual = regression_results["Real"] - regression_results["Classified"]
        residual_plot(residual, dataset_name)

if __name__ == "__main__" :
    main()
