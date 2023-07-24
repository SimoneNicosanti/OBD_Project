import pandas as pd
from Utils import *
from NeuralNetwork import NeuralNetwork


def main() :
    np.random.seed(123456)
    CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"

    dataset = pd.read_csv(CSV_URL) # tumore
    #dataset = pd.read_csv("./datasets/MNIST_Digits/train.csv") # immagini

    targetName = "malignant"
    #targetName = "label"

    #dataset = dataset.groupby("label").head(100)

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = dataset, targetName = targetName)

    featuresNumber = X_train.shape[1]
    labelsNumber = len(np.unique(Y_train))
    numberNeurons = int(2/3 * featuresNumber) + labelsNumber
    numberLayers = 2
    model = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons)

    model.fit(X_train, Y_train, max_steps = 500, epsilon = 1e-12)
    model.predict(X_test, Y_test)

if __name__ == "__main__" :
    main()