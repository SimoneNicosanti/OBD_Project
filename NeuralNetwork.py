import numpy as np
from Layer import Layer
from Utils import *

class NeuralNetwork :

    def __init__(self, hiddenLayerNum : int, inputDim : int, outputDim : int, neuronNum : int) -> None:

        self.hiddenLayerNum = hiddenLayerNum

        self.firstLayer = Layer(inputDim)
        self.lastLayer = Layer(outputDim)

        self.train_mean : np.ndarray = None
        self.train_std : float = 1
  
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


    def predict(self, X_test : np.ndarray, Y_test : np.ndarray) -> np.ndarray :
        numpyLabels : np.ndarray = np.array(Y_test)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        # TODO
        # inserire media del training set

        normalized_X_test = (X_test - self.train_mean) / self.train_std
        
        accuracy = 0

        for i in range(0, len(normalized_X_test)) :
            elem = normalized_X_test[i]
            self.do_forwarding(elem)

            #print("Output", self.lastLayer.getOutput())
            elemLabel = Y_test[i]

            elemLabelsArray : np.ndarray = (sortedLabels == elemLabel).astype(int)

            networkOutput = self.lastLayer.getOutput()
            #networkOutput = networkOutput / np.linalg.norm(networkOutput, 1)

            probs = softmax(networkOutput)
            maxProbIndex = probs.argmax()
            predictions = np.zeros(probs.shape)
            predictions[maxProbIndex] = 1

            realClassIndex = elemLabelsArray.argmax()
            if (predictions[realClassIndex] == 1) :
                accuracy += 1

            print(predictions, elemLabelsArray)
        accuracy /= X_test.shape[0]
        print(accuracy)

        return accuracy
    
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
                    de_dw.append(de_da[j] * da_dw)
                    layer.de_dw_bias[j] += de_da[j] * da_dw
                else:
                    da_dw = prev_layer.getOutput()[i]
                    de_dw.append(de_da[j] * da_dw)
                    layer.de_dw_matrix[i][j] += de_da[j] * da_dw

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
                        de_dw.append(de_da[j] * da_dw)
                        layer.de_dw_bias[j] += de_da[j] * da_dw
                    else:
                        da_dw = prev_layer.getOutput()[i]
                        de_dw.append(de_da[j] * da_dw)
                        layer.de_dw_matrix[i][j] += de_da[j] * da_dw

            layer = layer.prevLayer

        return np.array(de_dw)
    
    def update_weights(self, alpha) :
        layer : Layer = self.firstLayer.nextLayer
        #alpha = diminishing_stepsize(k)
        while layer != None :
            
            # print("weight matrix", layer.weightMatrix)
            # print("bias: ", layer.biasArray)
            # print("update matrix: ", layer.de_dw_matrix)
            # print("update bias: ", layer.de_dw_bias)

            layer.update_weights(alpha)
            layer = layer.nextLayer


    def reset_de_dw(self) -> None :
        layer : Layer = self.firstLayer.nextLayer
        while layer != None :
            layer.reset_de_dw()
            layer = layer.nextLayer


    def fit(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4) :

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0)
        normalized_X_train = (X_train - self.train_mean) / self.train_std
        

        ## TODO
        ## Y_train è un unico array: bisogna convertire il singolo y_train in un array di tutti 0 ed un 1 --> Fatto
        de_dw_tot : np.ndarray = None
        initialized = False
        precision = epsilon
        k = 0
        while (precision >= epsilon and k <= max_steps) :
            print("Precisione: ", precision, "--", "K: ", k)
            self.reset_de_dw()
            for i in range(0, len(normalized_X_train)) :
                elem = normalized_X_train[i]
                self.do_forwarding(elem)

                elemLabel = Y_train[i]
                elemLabelsArray = (sortedLabels == elemLabel).astype(int)

                de_dw : np.ndarray = self.backpropagation(elemLabelsArray)
                if (not initialized) :
                    de_dw_tot = de_dw
                    initialized = True
                else :
                    de_dw_tot += de_dw
            
            precision = np.linalg.norm(de_dw_tot)
            initialized = False
            self.update_weights(1 / precision)
            k += 1

        #print(de_dw_tot)
    