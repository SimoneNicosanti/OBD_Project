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
        self.do_forwarding(input)

        networkOutput = self.lastLayer.getOutput()

        probs = softmax(networkOutput)

        maxProbIndex = probs.argmax()

        predictions = np.zeros(probs.shape)
        predictions[maxProbIndex] = 1

        return predictions
    
    def do_forwarding(self, input) :
        layer : Layer = self.firstLayer
        layer.forwardPropagation(input)
        layer = layer.nextLayer

        ## TODO: Correggere i layer.nextLayer e i layer.prevLayer nei cicli 
        # messo così non si opera mai su prime e ultimo livello --> combiare in layer != None
        while layer != None:
            layer.forwardPropagation()
            layer = layer.nextLayer

        return 
    

    def backpropagation(self, labels : np.ndarray) -> np.ndarray :
        layer : Layer = self.lastLayer
        # TODO
        # controllare indici del for --> Forse risolto con +1
        de_da = []
        de_dw = []
        de_dy = derivative_e_y(layer.getOutput(), labels)
        for j in range(0, layer.neuronNumber) :
            prev_layer : Layer = layer.prevLayer
            de_da.append(de_dy[j] * 1)
            for i in range(0, prev_layer.neuronNumber + 1) :
                if (i == prev_layer.neuronNumber) :
                    da_dw = -1
                else:
                    da_dw = prev_layer.getOutput()[i]
                de_dw.append(de_da[j] * da_dw)

        print("de_da", de_da)

        layer = layer.prevLayer

        while layer.prevLayer != None :
            prev_layer : Layer = layer.prevLayer
            next_layer : Layer = layer.nextLayer
            de_da_prev : np.ndarray = np.array(de_da)
            de_da = []
            for j in range(0, layer.neuronNumber) :
                dg = layer.relu_derivative(layer.aArray[j])
                de_da.append(dg * np.dot(de_da_prev, next_layer.weightMatrix[j]))
                for i in range(0, prev_layer.neuronNumber + 1) :
                    if (i == prev_layer.neuronNumber) :
                        da_dw = -1
                    else:
                        da_dw = prev_layer.getOutput()[i]
                    de_dw.append(de_da[j] * da_dw)

            layer = layer.prevLayer

        return np.array(de_dw)


    def fit(self, X_train, Y_train) :

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        ## TODO
        ## Y_train è un unico array: bisogna convertire il singolo y_train in un array di tutti 0 ed un 1 --> Fatto
        de_dw_tot = None
        initialized = False
        for i in range(0, 1) :
            elem = X_train[i]
            self.do_forwarding(elem)

            print("Output", self.lastLayer.getOutput())
            elemLabel = Y_train[i]

            elemLabelsArray = (sortedLabels == elemLabel).astype(int)

            de_dw : np.ndarray = self.backpropagation(elemLabelsArray)
            if (not initialized) :
                de_dw_tot = de_dw
                initialized = True
            else :
                de_dw_tot += de_dw

        print(de_dw_tot)
    