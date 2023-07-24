from NeuralNetwork import NeuralNetwork
import numpy as np

# TODO : Cross Validation
def crossValidate(X_train, Y_train, X_valid, Y_valid) -> NeuralNetwork :

    lambdaArray = np.arange(start = 0.1, stop = 5, step = 0.5)
    layerNumArray = np.arange(start = 1, stop = 6, step = 1)
    
    for lam in lambdaArray :
        for layerNum in layerNumArray :
            neuronNumMatrix = generateNeuronNumConfigurations(layerNum)

    pass


def generateNeuronNumConfigurations(layerNum : int) :
    pass