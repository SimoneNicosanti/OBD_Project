import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def train_test_split(X, y, test_size=0.3, random_state=None):

  if(random_state!=None):
    np.random.seed(random_state)
  
  n = X.shape[0]

  test_indices = np.random.choice(n, int(n*test_size), replace=False) # selezioniamo gli indici degli esempi per il test set
  
  # estraiamo gli esempi del test set
  # in base agli indici
  
  X_test = X[test_indices]
  y_test = y[test_indices]
  
  # creiamo il train set
  # rimuovendo gli esempi del test set
  # in base agli indici
  
  X_train = np.delete(X, test_indices, axis=0)
  y_train = np.delete(y, test_indices, axis=0)

  return (X_train, X_test, y_train, y_test )

def datasetSplit(dataset : pd.DataFrame, targetName : str, targetDrop : list, targetOHE : list, testSetSize : float = 0.2, validationSetSize : float = 0.2) :
    pre_processed_dataset = dataset.drop(targetName, axis = 1)
    pre_processed_dataset = pre_processed_dataset.drop(targetDrop, axis = 1)
    pre_processed_dataset = oneHotEncoding(pre_processed_dataset, targetOHE)
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
  print(dataset)
  return dataset

# TODO : inserire Loss Function per la regressione
def softmax(output : np.ndarray) -> np.ndarray :
  expon = np.power(np.e, output - output.max())
  return expon / np.sum(expon)

def derivative_cross_entropy(output : np.ndarray) -> np.ndarray:
  return 1 - softmax(output)

def derivative_e_y(output : np.ndarray, labels : np.ndarray) -> np.ndarray:
  #print(- labels * (1 - softmax(output)))
  return softmax(output) - labels
  #return -1 + labels * softmax(output)
  #return - labels + softmax(output)
  #return np.dot(labels, derivative_cross_entropy(output))

# TODO : AdaGrad -> RMSProp -> Adadelta -> Adam -> Nadam
def adaGrad_stepsize() -> float:
  return 