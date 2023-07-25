import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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

def datasetSplit(dataset : pd.DataFrame, targetName : str, testSetSize : float = 0.2, validationSetSize : float = 0.2) :
    
    featuresMatrix = dataset.drop(targetName, axis=1).values 

    # stelle
    #featuresMatrix = dataset.drop(targetName, axis = 1)
    #featuresMatrix = featuresMatrix.drop("obj_ID", axis = 1) # TODO : rimuovere ID per la classificazione di stelle
    #featuresMatrix = featuresMatrix.drop("run_ID", axis = 1)
    #featuresMatrix = featuresMatrix.drop("rerun_ID", axis = 1)
    #featuresMatrix = featuresMatrix.drop("field_ID", axis = 1)
    #featuresMatrix = featuresMatrix.drop("spec_obj_ID", axis = 1)
    #featuresMatrix = featuresMatrix.drop("fiber_ID", axis = 1)
    #featuresMatrix = featuresMatrix.values

    # musica
    #featuresMatrix = dataset.drop(targetName, axis = 1)
    #featuresMatrix = (featuresMatrix.drop("Unnamed: 0", axis = 1)) # TODO : rimuovere Unnamed: 0 per la classificazione di canzoni
    #featuresMatrix = featuresMatrix.values

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

def oneHotEncoding() :
  data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
  values = np.array(data)
  print(values)
  # integer encode
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(values)
  print(integer_encoded)
  # binary encode
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  print(onehot_encoded)
  # invert first example
  inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
  print(inverted)

def cartesian_plot(x : list, y : list, x_label : str, y_label : str, title : str) -> None :
  plt.plot(x,y)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.show()
  return

def bar_plot(x : list, y : list, x_label : str, y_label : str, title : str) -> None :
  plt.bar(x, y, width = 0.6, color = ['red', 'black'])
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title)
  plt.show()
  return

def pie_plot(x: list, y : list) -> None :
  fig = plt.figure(figsize = (7, 7))
  plt.pie(x, labels = y, autopct = '%1.1f%%')
  plt.show()
  return

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