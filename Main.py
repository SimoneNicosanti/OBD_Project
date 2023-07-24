import pandas as pd
from Utils import *
from NeuralNetwork import NeuralNetwork


def main() :
    np.random.seed(123456)

    #--------------------------------------------TUMORE---------------------------------------------------------#
    #CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"
    #dataset = pd.read_csv(CSV_URL)
    #targetName = "malignant"
    #--------------------------------------------IMMAGINI-------------------------------------------------------#
    #aggiungere + 1 alla std_dev
    #dataset = pd.read_csv("./datasets/Classification_Digits/train.csv")
    #dataset = dataset.groupby("label").head(100)
    #targetName = "label"
    #--------------------------------------------PISTACCHI------------------------------------------------------#
    dataset = pd.read_csv("./datasets/Classification_Pistachio/pistachio.csv")
    targetName = "Class"
    #--------------------------------------------STELLE---------------------------------------------------------#
    #rimuovere ID come feature
    #dataset = pd.read_csv("./datasets/Classification_Stars/stars.csv")
    #targetName = "class"
    #--------------------------------------------MUSICA---------------------------------------------------------#
    #dataset = pd.read_csv("./datasets/Classification_Music/songs.csv")
    #targetName = "labels"
    #-----------------------------------------------------------------------------------------------------------#

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = dataset, targetName = targetName)

    featuresNumber = X_train.shape[1]
    labelsNumber = len(np.unique(Y_train))
    numberNeurons = int(2/3 * featuresNumber) + labelsNumber
    numberLayers = 2
    model = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons)

    print("Starting training with training set:")
    model.fit(X_train, Y_train, max_steps = 100, epsilon = 1e-12)

    accuracy_trainining = model.predict(X_train, Y_train)
    accuracy_generalization = model.predict(X_test, Y_test)
    print("Training Accuracy: ", accuracy_trainining)
    print("Generalization Accuracy:", accuracy_generalization)

    bar_plot(["Training Accuracy", "Generalization Accuracy"], [accuracy_trainining, accuracy_generalization], "Type of accuracy", "Accuracy", "Bar plot for accuracies")

    pie_plot([len(X_train), len(X_valid), len(X_test)], ["Training Set", "Validation Set", "Test Set"])

if __name__ == "__main__" :
    main()