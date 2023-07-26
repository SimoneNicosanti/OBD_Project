dataset_dict = {
    "Cancer" : {
        "fileName" : "./datasets/Classification_Cancer/train.csv",
        "targetName" : "malignant" ,
        "toDrop" : [],
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
        "classification" : True
        # dataset["Platform"] = dataset["Cabin"].str.slice(0,1)
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
    }
}