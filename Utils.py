import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

## Preelabora il dataset:
# - Esegue il OneHotEncoding su features categoriche
# - Rimuove righe aventi delle features nulle
# - Divide il dataset in Training, Validation e Test set, ritornando per ognuno la matrice delle features e il vettore dei target 
def datasetPreprocess(dataset : pd.DataFrame, targetName : str, targetDrop : list, targetOHE : list, testSetSize : float = 0.15, validationSetSize : float = 0.15) :

    pre_processed_dataset = dataset
    pre_processed_dataset = pre_processed_dataset.drop(targetDrop, axis = 1)
    pre_processed_dataset = oneHotEncoding(pre_processed_dataset, targetOHE)
    pre_processed_dataset = pre_processed_dataset.dropna(axis = 0)
    pre_processed_dataset = pre_processed_dataset.drop(targetName, axis = 1)

    featuresMatrix = pre_processed_dataset.values

    labelsColumn = dataset[targetName].values

    pointsNumber : int = featuresMatrix.shape[0]
    indexesArray : np.ndarray = np.arange(0, pointsNumber)
    np.random.shuffle(indexesArray)

    testPointsNum = int(len(indexesArray) * testSetSize)
    validPointsNum = int(len(indexesArray) * validationSetSize)

    testIndexes = indexesArray[0 : testPointsNum]
    validationIndexes = indexesArray[testPointsNum : testPointsNum + validPointsNum]
    trainIndexes = indexesArray[testPointsNum + validPointsNum : ]
    
    X_train, Y_train = featuresMatrix[trainIndexes], labelsColumn[trainIndexes]
    X_valid, Y_valid = featuresMatrix[validationIndexes], labelsColumn[validationIndexes]
    X_test, Y_test = featuresMatrix[testIndexes], labelsColumn[testIndexes]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

## Esegue il OneHotEncoding del dataset
# @dataset : DataFrame --> DataFrame di cui fare il OneHotEncoding
# @column_names : list --> Lista dei nomi delle colonne di cui fare OneHotEncoding
def oneHotEncoding(dataset : pd.DataFrame, column_names : list) -> pd.DataFrame :
    for column_name in column_names :
        encoder = OneHotEncoder(sparse=False)
        encoded = pd.DataFrame(encoder.fit_transform(dataset[[column_name]]))
        encoded.columns = encoder.get_feature_names_out([column_name])

        dataset.drop([column_name], axis = 1, inplace = True)
        dataset = pd.concat([dataset, encoded], axis = 1)
    return dataset


def squaredErrorFunction(output : np.ndarray, realValues : np.ndarray) -> np.ndarray :
    return (np.linalg.norm(output - realValues)) ** 2


def softmax(output : np.ndarray) -> np.ndarray :
    expon = np.power(np.e, output - output.max(axis = 1, keepdims = True))
    return expon / np.sum(expon, axis = 1, keepdims = True)


def derivative_e_y(output : np.ndarray, realValues : np.ndarray, isClassification : bool) -> np.ndarray:
    if (isClassification) :
        return np.squeeze(softmax(output) - realValues)
    else :
        return 2 * (output - realValues)


def middle_error(output : np.ndarray, realValuesMatrix : np.ndarray, isClassification : bool) -> float :
    if (isClassification) :
        cross_entropy = - (realValuesMatrix * np.log(softmax(output) + 1e-6))
        return np.sum(cross_entropy) / realValuesMatrix.shape[0]
    else :
        squared_error = np.linalg.norm(realValuesMatrix - output, axis = 1) ** 2
        return np.sum(squared_error) / realValuesMatrix.shape[0]
