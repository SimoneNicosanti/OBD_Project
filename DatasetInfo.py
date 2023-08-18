import pandas as pd

def diamonds_preprocess(diamondsDataset : pd.DataFrame) :
    diamondsDataset.clarity = diamondsDataset.clarity.replace({"I1" : 1, "SI2" : 2, "SI1" : 3, "VS2" : 4, "VS1" : 5, "VVS2" : 6, "VVS1" : 7, "IF" : 8})
    diamondsDataset.color = diamondsDataset.color.replace({"D" : 7, "E" : 6, "F" : 5, "G" : 4, "H" : 3, "I" : 2, "J" : 1})

def spaceship_preprocess(spaceshipDataset : pd.DataFrame) :
    spaceshipDataset["Platform"] = spaceshipDataset["Cabin"].str.slice(0,1)

dataset_dict = {
    
    "Cancer" : {
        "fileName" : "./datasets/Classification_Cancer/train.csv",
        "targetName" : "malignant" ,
        "toDrop" : ["Unnamed: 0"],
        "toOHE" : [] ,
        "classification" : True
        } ,
    
    "MNIST" : {
        "fileName" : "./datasets/Classification_Digits/train.csv",
        "targetName" : "label",
        "toDrop" : [] ,
        "toOHE" : [] ,
        "classification" : True
    } ,

    "Chinese" : {
        "fileName" : "./datasets/Classification_Chinese/compressed_dataset.csv",
        "targetName" : "label",
        "toDrop" : ["character"] ,
        "toOHE" : [] ,
        "classification" : True
    } ,

    
    "Pistachio" : {
        "fileName" : "./datasets/Classification_Pistachio/pistachio.csv",
        "targetName" : "Class" ,
        "toDrop" : [],
        "toOHE" : [] ,
        "classification" : True
    } ,

    "Stars" : {
        "fileName" : "./datasets/Classification_Stars/stars.csv",
        "targetName" : "class" ,
        "toDrop" : ["obj_ID", "run_ID", "rerun_ID", "field_ID", "fiber_ID", "spec_obj_ID"],
        "toOHE" : [] ,
        "classification" : True
    } ,

    "Songs" : {
        "fileName" : "./datasets/Classification_Music/songs.csv",
        "targetName" : "labels" ,
        "toDrop" : ["Unnamed: 0"],
        "toOHE" : [] ,
        "classification" : True
    } ,

    "Titanic" : {
        "fileName" : "./datasets/Classification_Titanic/Titanic.csv",
        "targetName" : "Survived" ,
        "toDrop" : ["PassengerId"],
        "toOHE" : ["Sex"] ,
        "classification" : True
    } ,

    "Spaceship" : {
        "fileName" : "./datasets/Classification_SpaceshipTitanic/SpaceshipTitanic.csv",
        "targetName" : "Transported" ,
        "toDrop" : ["PassengerId", "Cabin"],
        "toOHE" : ["HomePlanet", "CryoSleep", "Destination", "VIP", "Grouped", "Deck", "Side", "Has_expenses", "Is_Embryo", "Platform"] ,
        "classification" : True ,
        "preprocess_function" : spaceship_preprocess
    } ,

    "Fire" : {
        "fileName" : "./datasets/Classification_Fire/AcousticExtinguisherFire.csv",
        "targetName" : "class" ,
        "toDrop" : [],
        "toOHE" : ["fuel"] ,
        "classification" : True
    } ,

    "Students" : {
        "fileName" : "./datasets/Regression_Student/StudentPerformance.csv",
        "targetName" : "Performance Index" ,
        "toDrop" : [],
        "toOHE" : ["Extracurricular Activities"] ,
        "classification" : False
    } ,

    "Diamonds" : {
        "fileName" : "./datasets/Regression_Diamonds/train.csv",
        "targetName" : "price" ,
        "toDrop" : [],
        "toOHE" : ["cut"] ,
        "classification" : False ,
        "preprocess_function" : diamonds_preprocess
    } ,

    "House" : {
        "fileName" : "./datasets/Regression_House/HousePrice.csv",
        "targetName" : "median_house_value" ,
        "toDrop" : ["longitude", "latitude"],
        "toOHE" : ["ocean_proximity"] ,
        "classification" : False
    } ,

    "News" : {
        "fileName" : "./datasets/Regression_OnlineNews/OnlineNewsPopularity.csv",
        "targetName" : " shares" ,
        "toDrop" : ["url", " timedelta"],
        "toOHE" : [] ,
        "classification" : False
    } ,

    "Concrete" : {
        "fileName" : "./datasets/Regression_Concrete/Concrete.csv",
        "targetName" : "csMPa" ,
        "toDrop" : [],
        "toOHE" : [] ,
        "classification" : False
    } ,

    "Cars" : {
        "fileName" : "./datasets/Regression_Car/dataset.csv",
        "targetName" : "price" ,
        "toDrop" : ["ID", "name"],
        "toOHE" : ["fueltypes", "aspiration", "doornumbers", "carbody", "drivewheels", "enginelocation", "enginetype", "cylindernumber", "fuelsystem"] ,
        "classification" : False
    } ,

    "Paris" : {
        "fileName" : "./datasets/Regression_Paris/dataset.csv",
        "targetName" : "price" ,
        "toDrop" : [],
        "toOHE" : [] ,
        "classification" : False
    }
}

