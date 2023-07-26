import pandas as pd
from Utils import *
from LogWriter import *
from NeuralNetwork import NeuralNetwork


def main() :
    np.random.seed(123456)

    #--------------------------------------------TUMORE---------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Cancer/train.csv")
    # targetName = "malignant"
    # targetDrop = []
    # targetOHE = []
    #--------------------------------------------IMMAGINI-------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Digits/train.csv")
    # dataset = dataset.groupby("label").head(100)
    # targetName = "label"
    # targetDrop = []
    # targetOHE = []    
    #--------------------------------------------PISTACCHI------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Pistachio/pistachio.csv")
    # targetName = "Class"
    # targetDrop = []
    # targetOHE = []
    #--------------------------------------------STELLE---------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Stars/stars.csv")
    # targetName = "class"
    # targetDrop = ["obj_ID", "run_ID", "rerun_ID", "field_ID", "fiber_ID", "spec_obj_ID"]
    # targetOHE = []
    #--------------------------------------------MUSICA---------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Music/songs.csv")
    # targetName = "labels"
    # targetDrop = ["Unnamed: 0"]
    # targetOHE = []
    #--------------------------------------------TITANIC--------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Titanic/Titanic.csv")
    # targetName = "Survived"
    # targetDrop = ["PassengerId"]
    # targetOHE = ["Sex"]
    #--------------------------------------------SPACESHIP-TITANIC----------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_SpaceshipTitanic/SpaceshipTitanic.csv")
    # targetName = "Transported"
    # dataset["Platform"] = dataset["Cabin"].str.slice(0,1)
    # targetDrop = ["PassengerId", "Cabin"]
    # targetOHE = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Grouped", "Deck", "Side", "Has_expenses", "Is_Embryo", "Platform"]
    #--------------------------------------------FIRE-----------------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Classification_Fire/AcousticExtinguisherFire.csv")
    # targetName = "class"
    # targetDrop = []
    # targetOHE = ["fuel"]
    #--------------------------------------------STUDENT-PERFORMANCE--------------------------------------------#
    # dataset = pd.read_csv("./datasets/Regression_Student/StudentPerformance.csv")
    # targetName = "Performance Index"
    # targetDrop = []
    # targetOHE = ["Extracurricular Activities"]
    #--------------------------------------------HOUSE-PRICE----------------------------------------------------#
    # dataset = pd.read_csv("./datasets/Regression_House/HousePrice.csv")
    # targetName = "median_house_value"
    # targetDrop = ["longitude", "latitude"]
    # targetOHE = ["ocean_proximity"]
    #--------------------------------------------ONLINE-NEWS-POPULARITY-----------------------------------------#
    # dataset = pd.read_csv("./datasets/Regression_OnlineNews/OnlineNewsPopularity.csv")
    # targetName = "self_reference_avg_shares"
    # targetDrop = ["url", "timedelta"]
    # targetOHE = []

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = datasetSplit(dataset = dataset, targetName = targetName, targetDrop = targetDrop, targetOHE = targetOHE)

    featuresNumber = X_train.shape[1]
    labelsNumber = len(np.unique(Y_train))
    numberNeurons = int(2/3 * featuresNumber) + labelsNumber
    numberLayers = 2
    model = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification = False)

    print("Starting training with training set:")
    model.fit(X_train, Y_train, max_steps = 2000, epsilon = 1e-12, with_SAGA = True)

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