from Layer import Layer
import numpy as np

class Layer :

    def __init__(self, neuronNumber : int) -> None:
        self.neuronNumber = neuronNumber
        self.prevLayer = None
        self.nextLayer = None
        self.biasArray : np.ndarray = np.zeros(neuronNumber)
        self.outputArray : np.ndarray = np.zeros(neuronNumber)


    def setPrevAndNextLayer(self, prevLayer : Layer, nextLayer : Layer) :
        self.weightMatrix : np.ndarray = np.zeros((self.neuronNumber, nextLayer.getNeuronNumber()))
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

    def getNeuronNumber(self) -> int :
        return self.neuronNumber
    
    def forwardPropagation(self) -> None:
        inputArray = self.prevLayer.getOutput()
        z_array = np.dot(self.weightMatrix, inputArray) + self.biasArray
        self.outputArray = self.__relu(z_array)

        return
    
    def getOutput(self) -> np.ndarray :
        return self.outputArray
    

    def __relu(self, z_array) :
        zeroArray = np.zeros(self.neuronNumber)
        return np.maximum(z_array, zeroArray)
        