import numpy as np
from Layer import Layer
from LogWriter import *
from Utils import *

class NeuralNetwork :

    def __init__(self, hiddenLayerNum : int, inputDim : int, outputDim : int, neuronNum : int, isClassification : bool = True) -> None:

        self.isClassification : bool = isClassification
        self.hiddenLayerNum = hiddenLayerNum

        self.firstLayer = Layer(inputDim)
        if (self.isClassification) :
            self.lastLayer = Layer(outputDim)
        else :
            self.lastLayer = Layer(1)

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


    def predict(self, X_test : np.ndarray, Y_test : np.ndarray, logFileName : str) -> np.ndarray :
        numpyLabels : np.ndarray = np.array(Y_test)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        predictionArray = []

        normalized_X_test = (X_test - self.train_mean) / self.train_std
        
        accuracy = 0

        for i in range(0, len(normalized_X_test)) :
            elem = normalized_X_test[i]
            self.do_forwarding(elem)

            elemLabel = Y_test[i]

            networkOutput = self.lastLayer.getOutput()

            if (self.isClassification) :
                elemLabelsArray : np.ndarray = (sortedLabels == elemLabel).astype(int)
                probs = softmax(networkOutput)
                maxProbIndex = probs.argmax()
                predictions = np.zeros(probs.shape)
                predictions[maxProbIndex] = 1

                realClassIndex = elemLabelsArray.argmax()
                if (predictions[realClassIndex] == 1) :
                    accuracy += 1
            else :
                predictions = networkOutput
                elemLabelsArray = elemLabel
                accuracy += squaredErrorFunction(predictions, elemLabelsArray)

            predictionArray.append([predictions, elemLabelsArray])

        writeClassificationLog(logFileName, predictionArray)
        accuracy /= X_test.shape[0]

        return accuracy
    
    def do_forwarding(self, input) :
        layer : Layer = self.firstLayer
        layer.forwardPropagation(input)
        layer = layer.nextLayer

        while layer != None:
            layer.forwardPropagation()
            layer = layer.nextLayer

        return 

    # TODO : ricontrollare algoritmo ed eventualmente implementarla per tutto il training set 
    def backpropagation(self, labels : np.ndarray, point_index : int) -> np.ndarray :
        layer : Layer = self.lastLayer

        de_da = []
        de_dw = []
        de_dy = derivative_e_y(layer.getOutput()[point_index], labels, self.isClassification)
        for j in range(0, layer.neuronNumber) :
            prev_layer : Layer = layer.prevLayer
            de_da.append(de_dy[j] * 1)
            for i in range(0, prev_layer.neuronNumber + 1) :
                if (i == prev_layer.neuronNumber) :
                    da_dw = -1
                    de_dw.append(de_da[j] * da_dw)
                    layer.de_dw_bias[j] += de_da[j] * da_dw
                else:
                    da_dw = prev_layer.getOutput()[point_index][i]
                    de_dw.append(de_da[j] * da_dw)
                    layer.de_dw_matrix[i][j] += de_da[j] * da_dw

        
        layer = layer.prevLayer

        while layer.prevLayer != None :
            prev_layer : Layer = layer.prevLayer
            next_layer : Layer = layer.nextLayer
            de_da_prev : np.ndarray = np.array(de_da)
            de_da = []
            
            for j in range(0, layer.neuronNumber) :
                dg = layer.relu_derivative(layer.aArray[point_index][j])
                de_da.append(dg * np.dot(de_da_prev, next_layer.weightMatrix[j]))
                for i in range(0, prev_layer.neuronNumber + 1) :
                    if (i == prev_layer.neuronNumber) :
                        da_dw = -1
                        de_dw.append(de_da[j] * da_dw)
                        layer.de_dw_bias[j] += de_da[j] * da_dw
                    else:
                        da_dw = prev_layer.getOutput()[point_index][i]
                        de_dw.append(de_da[j] * da_dw)
                        layer.de_dw_matrix[i][j] += de_da[j] * da_dw

            layer = layer.prevLayer

        return np.array(de_dw)

    def reset_de_dw(self) -> None :
        layer : Layer = self.firstLayer.nextLayer
        while layer != None :
            layer.reset_de_dw()
            layer = layer.nextLayer


    def fit(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4, with_SAGA = False) -> None :
        if (with_SAGA) :
            self.__fit_saga(X_train, Y_train, epsilon, max_steps)
        else :
            self.__fit_dyn_sampl(X_train, Y_train, epsilon, max_steps)


    def __fit_dyn_sampl(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4) :

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std
        
        de_dw_tot : np.ndarray = None
        initialized_de_dw = False
        gradient_norm = epsilon
        gradient_norm_array = []
        k = 0
        while (gradient_norm >= epsilon and k <= max_steps) :
            self.reset_de_dw()

            # TODO : la letteratura dice che scegliere come dimensione del mini-batch un multiplo di 2 aiuta
            mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(int(1/25 * len(normalized_X_train) + k), len(normalized_X_train)))
            mini_batch_train = normalized_X_train[mini_batch_indexes]
            mini_batch_labels = Y_train[mini_batch_indexes]

            self.do_forwarding(mini_batch_train)

            for i in range(0, len(mini_batch_indexes)) :
                elemLabel = mini_batch_labels[i]
                if (self.isClassification) :
                    elemLabelsArray = (sortedLabels == elemLabel).astype(int)
                else :
                    elemLabelsArray = elemLabel

                de_dw : np.ndarray = self.backpropagation(elemLabelsArray, i)
                if (not initialized_de_dw) :
                    de_dw_tot = de_dw
                    initialized_de_dw = True
                else :
                    de_dw_tot += de_dw
            
            initialized_de_dw = False
            gradient_norm = np.linalg.norm(de_dw_tot)
            gradient_norm_array.append(gradient_norm)
            print("Gradient's norm: ", gradient_norm, "--", "K: ", k)
            k += 1

            self.rmsProp_update_weights(de_dw_tot)
            #self.update_weights(1 / gradient_norm)
            #self.update_weights(diminishing_stepsize(k))

        writeAllNormLog(gradient_norm_array)
        return
    

    def __fit_saga(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4) :

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std

        
        de_dw_tot : np.ndarray = None
        initialized_saga_acc = False
        gradient_norm = epsilon
        gradient_norm_array = []
        
        k = 0
        while (gradient_norm >= epsilon and k <= max_steps) :
            self.reset_de_dw()

            # TODO : la letteratura dice che scegliere come dimensione del mini-batch un multiplo di 2 aiuta
            mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(int(1/25 * len(normalized_X_train) + k), len(normalized_X_train)))
            #mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(256, len(normalized_X_train)))
            mini_batch_train = normalized_X_train[mini_batch_indexes]
            mini_batch_labels = Y_train[mini_batch_indexes]

            self.do_forwarding(mini_batch_train)

            sagaIndex = np.random.randint(0, len(mini_batch_indexes))

            elemLabel = mini_batch_labels[sagaIndex]
            if (self.isClassification) :
                elemLabelsArray = (sortedLabels == elemLabel).astype(int)
            else :
                elemLabelsArray = elemLabel

            de_dw : np.ndarray = self.backpropagation(elemLabelsArray, sagaIndex)
            
            if (not initialized_saga_acc) :
                ## TODO Chiedere al prof se è lecita l'inizializzazione randomica oppure se bisogna fare un'inizializzazione a 0
                #accumulatorSAGA = np.random.uniform(-1, 1, (X_train.shape[0], de_dw.shape[0]))
                accumulatorSAGA = np.zeros((X_train.shape[0], de_dw.shape[0]))
                initialized_saga_acc = True
            
            gradient_esteem = de_dw - (accumulatorSAGA[sagaIndex] - accumulatorSAGA.mean(axis = 0))
            
            accumulatorSAGA[sagaIndex] = de_dw

            gradient_norm = np.linalg.norm(gradient_esteem)
            gradient_norm_array.append(gradient_norm)
            print("Gradient's norm: ", gradient_norm, "--", "K: ", k)
            k += 1

            #self.saga_update_weights(gradient_esteem, diminishing_stepsize(k))
            self.rmsProp_update_weights(gradient_esteem)
            
        writeAllNormLog(gradient_norm_array)
        return

    def update_weights(self, alpha) :
        layer : Layer = self.firstLayer.nextLayer
        while layer != None :
            layer.update_weights(alpha)
            layer = layer.nextLayer

    def saga_update_weights(self, gradient_esteem : np.ndarray, alpha : float) -> None :
        layer = self.lastLayer
        i = 0
        start = 0
        while (layer.prevLayer != None) :
            layer.saga_update_weights(gradient_esteem, i, start, alpha)
            i += 1
            start = start + layer.neuronNumber * (layer.prevLayer.neuronNumber + 1)
            layer = layer.prevLayer
        return
    
    def adaGrad_update_weights(self, gradient_esteem : np.ndarray) -> None :
        layer = self.lastLayer
        i = 0
        start = 0
        while (layer.prevLayer != None) :
            layer.adaGrad_update_weights(gradient_esteem, start)
            i += 1
            start = start + layer.neuronNumber * (layer.prevLayer.neuronNumber + 1)
            layer = layer.prevLayer
        return
    
    def rmsProp_update_weights(self, gradient_esteem : np.ndarray) -> None :
        layer = self.lastLayer
        i = 0
        start = 0
        while (layer.prevLayer != None) :
            layer.rmsProp_update_weights(gradient_esteem, start)
            i += 1
            start = start + layer.neuronNumber * (layer.prevLayer.neuronNumber + 1)
            layer = layer.prevLayer
        return
