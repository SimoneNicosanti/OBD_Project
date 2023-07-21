import pandas as pd
from Utils import *
from NeuralNetwork import NeuralNetwork


def main() :
    np.random.seed(123456)
    CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"

    breast_cancer = pd.read_csv(CSV_URL)

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = breast_cancer, targetName = "malignant")
 
    model = NeuralNetwork(10, 30, 2, 128)

    model.fit(X_train, Y_train)
    # model.predict(X_test, Y_test)

if __name__ == "__main__" :
    main()