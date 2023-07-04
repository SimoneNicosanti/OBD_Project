import numpy as np

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