import numpy as np
from Layer import Layer
from Utils import *

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


    def predict(self, input : np.ndarray) -> np.ndarray :
        layer : Layer = self.firstLayer
        layer.forwardPropagation(input)
        layer = layer.nextLayer

        while (layer.nextLayer != None):
            layer.forwardPropagation()
            layer = layer.nextLayer

        prob = layer.getOutput()

        print(prob.shape)

        y = np.zeros(prob.shape)
        y[prob >= 0.5] = 1
        y[prob < 0.5] = 0

        return y

    def backpropagation(self, labels : np.ndarray) -> np.ndarray :
        layer : Layer = self.lastLayer
        # TODO
        # controllare indici del for
        de_da = []
        de_dw = []
        for j in range(0, layer.neuronNumber) :
            prev_layer : Layer = layer.prevLayer
            de_da.append(derivative_e_y(layer.getOutput()[j], labels[j]) * 1)
            for i in range(0, prev_layer.neuronNumber) :
                if (i == 0) :
                    da_dw = -1
                else:
                    da_dw = prev_layer.getOutput()[i]
                de_dw.append(de_da[j] * da_dw)

        layer = layer.prevLayer

        while layer.prevLayer != None :
            prev_layer : Layer = layer.prevLayer
            next_layer : Layer = layer.nextLayer
            de_da_prev : np.ndarray = np.array(de_da)
            de_da = []
            for j in range(0, prev_layer.neuronNumber) :
                dg = layer.relu_derivative(layer.aArray[j])
                de_da.append(dg * np.dot(de_da_prev, next_layer.weightMatrix[j]))
                for i in range(0, prev_layer.neuronNumber) :
                    if (i == 0) :
                        da_dw = -1
                    else:
                        da_dw = prev_layer.getOutput()[i]
                    de_dw.append(de_da[j] * da_dw)

            layer = layer.prevLayer

        return np.array(de_dw)

    def fit(self, x, y) :
        predictions : np.ndarray = self.predict(x)
        de_dw : np.ndarray = self.backpropagation(y)
        print(de_dw)