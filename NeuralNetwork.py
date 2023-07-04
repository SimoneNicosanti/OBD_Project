import numpy as np
from Layer import Layer

class NeuralNetwork :

    def __init__(self, hiddenLayerNum : int, inputDim : int, outputDim : int, neuronNum : int) -> None:
        self.hiddenLayerNum = hiddenLayerNum

        self.firstLayer = Layer(inputDim)
        self.lastLayer = Layer(outputDim)
  
        prevLayer = None
        currLayer = self.firstLayer
        for i in range(0, hiddenLayerNum + 2) :

            if (i == hiddenLayerNum) :
                nextLayer = self.lastLayer
            elif (i == hiddenLayerNum + 1) :
                nextLayer = None
            else :
                nextLayer = Layer(neuronNum)

            currLayer.setPrevAndNextLayer(prevLayer, nextLayer)
            prevLayer = currLayer
            currLayer = nextLayer

        return


    def evaluate(self, input : np.ndarray) -> np.ndarray :
        layer : Layer = self.firstLayer
        layer.forwardPropagation(input)
        layer = layer.nextLayer

        while (layer.nextLayer != None):
            layer.forwardPropagation()
            layer = layer.nextLayer

        prob = layer.getOutput()

        y = np.zeros(input.shape[0])
        y[prob >= 0.5] = 1
        y[prob < 0.5] = 0

        return y

    def fit() :
        pass

    def evaluate() :
        pass