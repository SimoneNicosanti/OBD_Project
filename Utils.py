import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def datasetSplit(dataset : pd.DataFrame, targetName : str, targetDrop : list, targetOHE : list, testSetSize : float = 0.2, validationSetSize : float = 0.2) :
    pre_processed_dataset = dataset.drop(targetName, axis = 1)
    pre_processed_dataset = pre_processed_dataset.drop(targetDrop, axis = 1)
    pre_processed_dataset = oneHotEncoding(pre_processed_dataset, targetOHE)
    pre_processed_dataset = pre_processed_dataset.dropna(axis = 0)
    
    featuresMatrix = pre_processed_dataset.values

    labelsColumn = dataset[targetName].values

    pointsNumber : int = featuresMatrix.shape[0]

    indexesArray = np.arange(start = 0, stop = pointsNumber)
    testIndexes = np.random.choice(indexesArray, int(pointsNumber * testSetSize), replace = False)

    indexesArray = np.setdiff1d(indexesArray, testIndexes)
    validationIndexes = np.random.choice(indexesArray, int(pointsNumber * validationSetSize), replace = False)

    indexesArray = np.setdiff1d(indexesArray, validationIndexes)
    trainIndexes = indexesArray

    X_train, Y_train = featuresMatrix[trainIndexes], labelsColumn[trainIndexes]
    X_valid, Y_valid = featuresMatrix[validationIndexes], labelsColumn[validationIndexes]
    X_test, Y_test = featuresMatrix[testIndexes], labelsColumn[testIndexes]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

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

def diminishing_stepsize(k : int) -> float :
    return 0.001 / (k + 1)