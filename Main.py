import pandas as pd
from Utils import *
from NeuralNetwork import NeuralNetwork


def main() :
    np.random.seed(123456)
    CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"

    dataset = pd.read_csv(CSV_URL)
    #dataset = pd.read_csv("./datasets/MNIST_Digits/train.csv")

    targetName = "malignant"

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = dataset, targetName = targetName)

    featuresNumber = X_train.shape[1]
    labelsNumber = len(np.unique(Y_train))
    model = NeuralNetwork(1, featuresNumber, labelsNumber, 128)

    model.fit(X_train, Y_train, max_steps = 500, epsilon = 0)
    model.predict(X_test, Y_test)

if __name__ == "__main__" :
    main()