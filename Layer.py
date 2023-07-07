import numpy as np

class Layer :

    def __init__(self, neuronNumber : int) -> None :
        self.neuronNumber = neuronNumber
        self.prevLayer = None
        self.nextLayer = None
        self.weightMatrix : np.ndarray = None
        self.biasArray : np.ndarray = np.zeros(neuronNumber)
        self.aArray : np.ndarray = np.zeros(neuronNumber)
        self.outputArray : np.ndarray = np.zeros(neuronNumber)

    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer
        
        if (prevLayer != None) :
            self.weightMatrix = np.random.uniform(-1, 1, ((self.prevLayer.getNeuronNumber(), self.getNeuronNumber())))
            self.biasArray = np.random.uniform(-1, 1, self.getNeuronNumber())

    def getNeuronNumber(self) -> int :
        return self.neuronNumber

    def forwardPropagation(self, input : np.ndarray = None) -> None :
        if (self.prevLayer == None) :
            self.outputArray = input
            #print("dim input: ", input.shape)
        else :
            inputArray = self.prevLayer.getOutput()
            #print("dim z: ", inputArray.shape)
            #print("dim w: ", self.weightMatrix.shape)
            #print("dim b:", self.biasArray.shape)

            # TODO
            # check dimensioni
            # print(self.weightMatrix)
            # print(inputArray)
            self.aArray = np.dot(self.weightMatrix.T, inputArray) + self.biasArray

            if (self.nextLayer == None) :
                #self.outputArray = self.__sigmoid(self.aArray)
                self.outputArray = self.aArray ## Identity function
                print("LastLayerOutput", self.outputArray)
            else :
                self.outputArray = self.__relu(self.aArray)

        return

    def getOutput(self) -> np.ndarray :
        return self.outputArray

    
    def __cross_entropy_loss(self, y_true, y_pred) :
        return -np.sum(y_true * np.log(y_pred))

    def __sigmoid(self, z_array) -> np.ndarray :
        return 1 / (1 + np.power(np.e, -z_array))

    def __relu(self, z_array) -> np.ndarray :
        return np.maximum(z_array, 0)
    
    def relu_derivative(self, z_array : np.ndarray) :
        dz : np.ndarray = np.zeros(z_array.shape)
        dz[z_array > 0] = 1
        return dz
        