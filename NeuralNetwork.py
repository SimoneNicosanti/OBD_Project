import numpy as np
from Layer import *
from LogWriter import *
from Utils import *
from StepEnum import *

class NeuralNetwork :

    def __init__(self, hiddenLayerNum : int, inputDim : int, outputDim : int, neuronNumArray : np.ndarray, isClassification : bool = True, method : StepEnum = StepEnum.RMSPROP) -> None:
    
        self.isClassification : bool = isClassification
        self.hiddenLayerNum : int = hiddenLayerNum

        self.firstLayer : Layer = self.__switch(inputDim, method)
        self.lastLayer : Layer = self.__switch(outputDim, method)

        self.train_mean : np.ndarray = None
        self.train_std : float = 1

        self.train_y_mean : float = 0.0
        self.train_y_std : float = 1.0
  
        prevLayer : Layer = None
        currLayer : Layer = self.firstLayer
        for i in range(0, hiddenLayerNum + 2) :

            if (i == hiddenLayerNum) :
                nextLayer = self.lastLayer
            elif (i == hiddenLayerNum + 1) :
                nextLayer = None
            else :
                nextLayer = self.__switch(neuronNumArray[i], method)

            currLayer.setPrevAndNextLayer(prevLayer, nextLayer)
            prevLayer = currLayer
            currLayer = nextLayer

        return

    def __switch(self, neuronNum : int, method : StepEnum) -> Layer :
        if method == StepEnum.ADAGRAD :
            return AdaGradLayer(neuronNum)
        elif method == StepEnum.RMSPROP :
            return RMSPropLayer(neuronNum)
        elif method == StepEnum.ADAM :
            return AdamLayer(neuronNum)
        elif method == StepEnum.NADAM :
            return NadamLayer(neuronNum)
        elif method == StepEnum.ADADELTA : 
            return AdadeltaLayer(neuronNum)
        else :
            return Layer(neuronNum)
        
    
    def predict_2(self, X_test : np.ndarray, Y_test : np.ndarray) -> np.ndarray :
        numpyLabels : np.ndarray = np.array(Y_test)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        predictionArray = []

        normalized_X_test = (X_test - self.train_mean) / self.train_std

        ## TODO Controllare per concorrenza
        if (not self.isClassification) :
            normalized_Y_test = (Y_test - self.train_y_mean) / self.train_y_std
        else :
            normalized_Y_test = Y_test
        
        accuracy = 0

        for i in range(0, len(normalized_X_test)) :
            elem = normalized_X_test[i][np.newaxis, :]
            self.do_forwarding(elem)

            elemLabel = normalized_Y_test[i]

            networkOutput = self.lastLayer.getOutput()

            if (self.isClassification) :
                elemLabelsArray : np.ndarray = (sortedLabels == elemLabel).astype(int)
                probs = softmax(networkOutput)
                maxProbIndex = probs.argmax()
                predictions = np.zeros(probs.shape[1])
                predictions[maxProbIndex] = 1
                

                realClassIndex = elemLabelsArray.argmax()
                if (predictions[realClassIndex] == 1) :
                    accuracy += 1
            else :
                predictions = networkOutput * self.train_y_std + self.train_y_mean
                elemLabelsArray = elemLabel * self.train_y_std + self.train_y_mean
                accuracy += squaredErrorFunction(predictions, elemLabelsArray)

            predictionArray.append([predictions, elemLabelsArray])

        accuracy /= X_test.shape[0]

        return accuracy, predictionArray
        

    def predict(self, X_test : np.ndarray, Y_test : np.ndarray) -> np.ndarray :
        numpyLabels : np.ndarray = np.array(Y_test)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        predictionArray = []

        normalized_X_test = (X_test - self.train_mean) / self.train_std

        ## TODO Controllare per concorrenza
        if (not self.isClassification) :
            normalized_Y_test = (Y_test - self.train_y_mean) / self.train_y_std
        else :
            normalized_Y_test = Y_test
        
        accuracy = 0

        for i in range(0, len(normalized_X_test)) :
            elem = normalized_X_test[i]
            self.do_forwarding(elem)

            elemLabel = normalized_Y_test[i]

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
                predictions = networkOutput * self.train_y_std + self.train_y_mean
                elemLabelsArray = elemLabel * self.train_y_std + self.train_y_mean
                accuracy += squaredErrorFunction(predictions, elemLabelsArray)

            predictionArray.append([predictions, elemLabelsArray])

        accuracy /= X_test.shape[0]

        return accuracy, predictionArray
    
    def do_forwarding(self, input) :
        layer : Layer = self.firstLayer
        layer.forwardPropagation(input)
        layer = layer.nextLayer

        while layer != None:
            layer.forwardPropagation()
            layer = layer.nextLayer

        return 
    
    def backpropagation_2(self, realValuesMatrix : np.ndarray, k : int) -> float :

        de_da_list : list[np.ndarray] = [None] * (self.hiddenLayerNum + 2)
        de_da_list[-1] = derivative_e_y_2(self.lastLayer.getOutput(), realValuesMatrix, self.isClassification) * 1 ## Controllato: uguale ad altro caso
        
        currLayer : Layer = self.lastLayer.prevLayer
        i = self.hiddenLayerNum
        while (currLayer.prevLayer != None) :
            de_da_list[i] = currLayer.relu_derivative(currLayer.aArray) * np.dot(currLayer.nextLayer.weightMatrix, de_da_list[i + 1].T).T
            currLayer = currLayer.prevLayer
            i -= 1

        currLayer = self.firstLayer.nextLayer
        i = 1
        gradientSquaredNormArray = []
        while (currLayer != None) :
            prevLayer : Layer = currLayer.prevLayer
            de_da = de_da_list[i]
            da_dw = np.append(prevLayer.getOutput().T, [[-1] * prevLayer.getOutput().shape[0]], axis = 0).T
            row_wise_outer = np.array([np.outer(de_da[r], da_dw[r]).reshape(-1) for r in range(0, da_dw.shape[0])])

            dr_dw = row_wise_outer.sum(axis = 0)
            gradientSquaredNormArray.append(np.linalg.norm(dr_dw) ** 2)
            
            currLayer.update_weights(dr_dw, 0, k)

            currLayer = currLayer.nextLayer
            i += 1


        return np.sum(gradientSquaredNormArray)

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
                else:
                    da_dw = prev_layer.getOutput()[point_index][i]
                    de_dw.append(de_da[j] * da_dw)
        
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
                    else:
                        da_dw = prev_layer.getOutput()[point_index][i]
                        de_dw.append(de_da[j] * da_dw)

            layer = layer.prevLayer

        return np.array(de_dw)

    def fit(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4, with_SAGA = False) -> tuple[list, list] :

        if (with_SAGA) :
            return self.__fit_saga(X_train, Y_train, epsilon, max_steps)
        else :
            #return self.__fit_dyn_sample(X_train, Y_train, epsilon, max_steps)
            return self.__fit_dyn_sample_2(X_train, Y_train, epsilon, max_steps)
        

    def __fit_dyn_sample_2(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4) -> tuple[list, list] :

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std

        if (self.isClassification) :
            normalized_Y_train = Y_train

            sortedLabelsMatrix = np.tile(sortedLabels, (X_train.shape[0], 1))
            labelsMatrix = np.tile(Y_train.T, (sortedLabels.shape[0], 1)).T
            realValuesMatrix = (sortedLabelsMatrix == labelsMatrix).astype(int)
        else :
            self.train_y_mean = Y_train.mean(axis = 0)
            self.train_y_std = Y_train.std(axis = 0)
            normalized_Y_train = (Y_train - self.train_y_mean) / self.train_y_std

            realValuesMatrix = normalized_Y_train[:, np.newaxis]
            
        
        de_dw_tot : np.ndarray = None
        initialized_de_dw = False
        gradient_norm = epsilon
        gradient_norm_array = []
        error_array = []
        k = 1
        while (gradient_norm >= epsilon and k <= max_steps) :
            ## TODO : Aggiungere l'epoca come il numero di volte in cui si sono visti tutti i campioni almeno una volta
            # TODO : la letteratura dice che scegliere come dimensione del mini-batch un multiplo di 2 aiuta

            mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(int(1/25 * len(normalized_X_train) + k), len(normalized_X_train)))
            mini_batch_train = normalized_X_train[mini_batch_indexes]
            mini_batch_labels = normalized_Y_train[mini_batch_indexes]

            self.do_forwarding(mini_batch_train)

            batchRealValuesMatrix = realValuesMatrix[mini_batch_indexes]
                
            gradientSquaredNorm = self.backpropagation_2(batchRealValuesMatrix, k)
            gradient_norm = np.sqrt(gradientSquaredNorm)

            self.do_forwarding(normalized_X_train)

            if (self.isClassification) :
                error = middle_error(self.lastLayer.getOutput(), realValuesMatrix, self.isClassification)
            else :
                output = self.lastLayer.getOutput() * self.train_y_std + self.train_y_mean
                matrix = realValuesMatrix * self.train_y_std + self.train_y_mean
                error = middle_error(output, matrix, self.isClassification)
                
            error_array.append(error)
            print("Gradient's norm: ", gradient_norm, "--", "Error: ", error, "--", "K: ", k)
            gradient_norm_array.append(gradient_norm)

            k += 1

        ## TODO Attenzione concorrenza dei thread
        #writeAllNormLog(gradient_norm_array)
        return gradient_norm_array, error_array

    def __fit_dyn_sample(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4) -> list :

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std

        if (not self.isClassification) :
            self.train_y_mean = Y_train.mean(axis = 0)
            self.train_y_std = Y_train.std(axis = 0)
            normalized_Y_train = (Y_train - self.train_y_mean) / self.train_y_std
        else :
            normalized_Y_train = Y_train
        
        de_dw_tot : np.ndarray = None
        initialized_de_dw = False
        gradient_norm = epsilon
        gradient_norm_array = []
        k = 0
        while (gradient_norm >= epsilon and k <= max_steps) :

            mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(int(1/25 * len(normalized_X_train) + k), len(normalized_X_train)))
            #mini_batch_indexes = np.random.randint(0, len(normalized_X_train), 1)
            mini_batch_train = normalized_X_train[mini_batch_indexes]
            mini_batch_labels = normalized_Y_train[mini_batch_indexes]

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

            self.update_weights(de_dw_tot, k)

        #writeAllNormLog(gradient_norm_array)
        return gradient_norm_array
    

    def __fit_saga(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, max_steps = 1e4) -> list :
        ## TODO Scrivere una versione di SAGA che vada bene per la backpropagation fatta tutta insieme

        numpyLabels : np.ndarray = np.array(Y_train)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std

        if (not self.isClassification) :
            self.train_y_mean = Y_train.mean(axis = 0)
            self.train_y_std = Y_train.std(axis = 0)
            normalized_Y_train = (Y_train - self.train_y_mean) / self.train_y_std
        else :
            normalized_Y_train = Y_train
        
        initialized_saga_acc = False
        gradient_norm = epsilon
        gradient_norm_array = []
        
        k = 0
        while (gradient_norm >= epsilon and k <= max_steps) :

            mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(int(1/25 * len(normalized_X_train) + k), len(normalized_X_train)))
            mini_batch_train = normalized_X_train[mini_batch_indexes]
            mini_batch_labels = normalized_Y_train[mini_batch_indexes]

            self.do_forwarding(mini_batch_train)

            sagaIndex = np.random.randint(0, len(mini_batch_indexes))

            # mini_batch_indexes = np.random.randint(0, len(normalized_X_train), min(int(1/25 * len(normalized_X_train) + k), len(normalized_X_train)))
            # mini_batch_train = normalized_X_train
            # mini_batch_labels = normalized_Y_train

            self.do_forwarding(mini_batch_train)

            sagaIndex = np.random.randint(0, len(mini_batch_indexes))

            elemLabel = mini_batch_labels[sagaIndex]
            if (self.isClassification) :
                elemLabelsArray = (sortedLabels == elemLabel).astype(int)
            else :
                elemLabelsArray = elemLabel

            de_dw : np.ndarray = self.backpropagation(elemLabelsArray, sagaIndex)
            
            if (not initialized_saga_acc) :
                accumulatorSAGA = np.zeros((X_train.shape[0], de_dw.shape[0]))
                initialized_saga_acc = True
            
            gradient_esteem = de_dw - (accumulatorSAGA[sagaIndex] - accumulatorSAGA.mean(axis = 0))
            
            accumulatorSAGA[sagaIndex] = de_dw

            gradient_norm = np.linalg.norm(gradient_esteem)
            gradient_norm_array.append(gradient_norm)
            print("Gradient's norm: ", gradient_norm, "--", "K: ", k)
            k += 1

            self.update_weights(gradient_esteem, k)

        #writeAllNormLog(gradient_norm_array)
        return gradient_norm_array

    def update_weights(self, gradient_esteem : np.ndarray, k : int) -> None :
        layer = self.lastLayer
        i = 0
        start = 0
        while (layer.prevLayer != None) :
            layer.update_weights(gradient_esteem, start, k)
            i += 1
            start = start + layer.neuronNumber * (layer.prevLayer.neuronNumber + 1)
            layer = layer.prevLayer
        return