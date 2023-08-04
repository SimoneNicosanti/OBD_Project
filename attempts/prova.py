import numpy as np
import pandas as pd

class NeuralNetwork:
  
  
  def __init__(self, hidden_layer_size=100):
    
    self.hidden_layer_size=hidden_layer_size
    
    
  def _init_weights(self, input_size, hidden_size):
    
    self._W1 = np.random.randn(input_size, hidden_size)
    self._b1 = np.zeros(hidden_size)
    self._W2 = np.random.randn(hidden_size,1)
    self._b2 = np.zeros(1)

    
  def _accuracy(self, y, y_pred):      
    return np.sum(y==y_pred)/len(y)
  
  
  def _log_loss(self, y_true, y_proba):
    return -np.sum(np.multiply(y_true,np.log(y_proba))+np.multiply((1-y_true),np.log(1-y_proba)))/len(y_true)
  
  
  def _relu(self, Z):
    return np.maximum(Z, 0)

  
  def _sigmoid(self, Z):
    return 1/(1+np.power(np.e,-Z))
  
  
  def _relu_derivative(self, Z):
    dZ = np.zeros(Z.shape)
    dZ[Z>0] = 1
    return dZ
    
               
  def _forward_propagation(self, X):
                     
    Z1 = np.dot(X,self._W1)+self._b1

    A1 = self._relu(Z1)
    Z2 = np.dot(A1,self._W2)+self._b2
    A2 = self._sigmoid(Z2)
    
    self._forward_cache = (Z1, A1, Z2, A2)

    return A2.ravel()


  def predict(self, X, return_proba=False):

      proba = self._forward_propagation(X)

      y = np.zeros(X.shape[0])
      y[proba>=0.5]=1
      y[proba<0.5]=0

      if(return_proba):
        return (y, proba)
      else:
        return proba
                            
      
  def _back_propagation(self, X, y):
  
    Z1, A1, Z2, A2 = self._forward_cache
                   
    m = A1.shape[1]
    
    dZ2 = A2-y.reshape(-1,1)
    dW2 = np.dot(A1.T, dZ2)/m
    db2 = np.sum(dZ2, axis=0)/m

    dZ1 = np.dot(dZ2, self._W2.T)*self._relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)/m
    db1 = np.sum(dZ1, axis=0)/m
    
    return dW1, db1, dW2, db2
           
               
  def fit(self, X, y, epochs=200, lr=0.01):
     
    self._init_weights(X.shape[1], self.hidden_layer_size)
      
    for _ in range(epochs):
      Y = self._forward_propagation(X)
      dW1, db1, dW2, db2 = self._back_propagation(X, y)
      self._W1-=lr*dW1
      self._b1-=lr*db1
      self._W2-=lr*dW2
      self._b2-=lr*db2
               

  def evaluate(self, X, y):
    y_pred, proba = self.predict(X, return_proba=True)
    accuracy = self._accuracy(y, y_pred)
    log_loss = self._log_loss(y, proba)
    return (accuracy, log_loss)


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




def main() :

    CSV_URL = "./datasets/Classification_Cancer/train.csv"

    breast_cancer = pd.read_csv(CSV_URL)
    X = breast_cancer.drop("malignant", axis=1).values
    y = breast_cancer["malignant"].values

    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3)

    X_max = X_train.max(axis=0)
    X_min = X_train.min(axis=0)

    X_train = (X_train - X_min)/(X_max-X_min)
    X_test = (X_test - X_min)/(X_max-X_min)

    model = NeuralNetwork()
    model.fit(X_train, y_train, epochs=500, lr=0.01)
    print(model.evaluate(X_test, y_test))


if __name__ == "__main__" :
  main()