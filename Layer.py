from Layer import Layer
import numpy as np

class Layer :

    def __init__(self, neuronNumber : int) -> None :
        self.neuronNumber = neuronNumber
        self.prevLayer : Layer = None
        self.nextLayer : Layer = None
        self.weightMatrix : np.ndarray = None
        self.biasArray : np.ndarray = np.zeros(neuronNumber)
        self.outputArray : np.ndarray = np.zeros(neuronNumber)

    def setPrevAndNextLayer(self, prevLayer : Layer, nextLayer : Layer) -> None :
        self.weightMatrix = np.random.uniform(-1, 1, ((self.neuronNumber, nextLayer.getNeuronNumber())))
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

    def getNeuronNumber(self) -> int :
        return self.neuronNumber

    def forwardPropagation(self, input : np.ndarray = None) -> None :
        if (self.prevLayer == None) :
            inputArray = input
        else :
            inputArray = self.prevLayer.getOutput()

        z_array = np.dot(self.weightMatrix, inputArray) + self.biasArray

        if (self.nextLayer == None) :
            self.outputArray = self.__sigmoid(z_array)
        else :
            self.outputArray = self.__relu(z_array)

        return
    
    def getOutput(self) -> np.ndarray :
        return self.outputArray

    def __sigmoid(self, z_array) -> np.ndarray :
        return 1 / (1 + np.power(np.e, -z_array))

    def __relu(self, z_array) -> np.ndarray :
        return np.maximum(z_array, 0)
    
    def _relu_derivative(self, z_array : np.ndarray):
        dz : np.ndarray = np.zeros(z_array.shape)
        dz[z_array > 0] = 1
        return dz
        