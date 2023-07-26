import pandas as pd
from Utils import *
from LogWriter import *
from NeuralNetwork import NeuralNetwork
from DatasetInfo import dataset_dict



def main() :
    np.random.seed(123456)

    dataset_info = dataset_dict["Songs"]
    dataset = pd.read_csv(dataset_info["fileName"])
    targetName = dataset_info["targetName"]
    toDrop = dataset_info["toDrop"]
    toOHE = dataset_info["toOHE"]
    isClassification = dataset_info["classification"]
    

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = dataset, targetName = targetName, targetDrop = toDrop, targetOHE = toOHE)

    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1
    numberNeurons = int(2/3 * featuresNumber) + labelsNumber
    #numberNeurons = 128
        
    numberLayers = 2
    model = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification = isClassification)

    print("Starting training with training set:")
    model.fit(X_train, Y_train, max_steps = 5000, epsilon = 1e-12, with_SAGA = True)

    accuracy_trainining = model.predict(X_train, Y_train, "training")
    accuracy_generalization = model.predict(X_test, Y_test, "generalization")
    print("Training Accuracy: ", accuracy_trainining)
    print("Generalization Accuracy:", accuracy_generalization)

    # normDataFrame = pd.read_csv("./log/NormLog.csv")
    # cartesian_plot(normDataFrame["K"], normDataFrame["Norm"], "Numero di iterazioni", "Norma del gradiente", "Norma del gradiente in funzione del numero di iterazioni")
    # bar_plot(["Training Accuracy", "Generalization Accuracy"], [accuracy_trainining, accuracy_generalization], "Type of accuracy", "Accuracy", "Bar plot for accuracies")
    # pie_plot([len(X_train), len(X_valid), len(X_test)], ["Training Set", "Validation Set", "Test Set"], "Ripartizione dataset")

if __name__ == "__main__" :
    main()