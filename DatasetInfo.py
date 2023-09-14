
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

    "Students" : {
        "fileName" : "./datasets/Regression_Student/StudentPerformance.csv",
        "targetName" : "Performance Index" ,
        "toDrop" : [],
        "toOHE" : ["Extracurricular Activities"] ,
        "classification" : False
    } ,

    "Concrete" : {
        "fileName" : "./datasets/Regression_Concrete/Concrete.csv",
        "targetName" : "csMPa" ,
        "toDrop" : [],
        "toOHE" : [] ,
        "classification" : False
    }
}

