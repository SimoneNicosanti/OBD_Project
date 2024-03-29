import numpy as np
from Layer import *
from LogWriter import *
from Utils import *
from StepEnum import *
import time

class NeuralNetwork :

    def __init__(self, hiddenLayerNum : int, inputDim : int, outputDim : int, neuronNumArray : np.ndarray, isClassification : bool = True, method : StepEnum = StepEnum.RMSPROP, lambdaL1 : float = 0, lambdaL2 : float = 0.1) -> None:

        self.neuronNumArray = neuronNumArray
        self.lambdaL1 = lambdaL1
        self.lambdaL2 = lambdaL2
        self.isClassification : bool = isClassification
        self.hiddenLayerNum : int = hiddenLayerNum

        self.firstLayer : Layer = self.__switch(inputDim, method, lambdaL1, lambdaL2)
        self.lastLayer : Layer = self.__switch(outputDim, method, lambdaL1, lambdaL2)

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
                nextLayer = self.__switch(neuronNumArray[i], method, lambdaL1, lambdaL2)

            currLayer.setPrevAndNextLayer(prevLayer, nextLayer)
            prevLayer = currLayer
            currLayer = nextLayer

        return

    def __switch(self, neuronNum : int, method : StepEnum, lambdaL1 : float, lambdaL2 : float) -> Layer :
        if method == StepEnum.ADAGRAD :
            return AdaGradLayer(neuronNum, lambdaL1, lambdaL2)
        elif method == StepEnum.RMSPROP :
            return RMSPropLayer(neuronNum, lambdaL1, lambdaL2)
        elif method == StepEnum.ADAM :
            return AdamLayer(neuronNum, lambdaL1, lambdaL2)
        elif method == StepEnum.NADAM :
            return NadamLayer(neuronNum, lambdaL1, lambdaL2)
        elif method == StepEnum.ADADELTA : 
            return AdaDeltaLayer(neuronNum, lambdaL1, lambdaL2)
        else :
            return Layer(neuronNum, lambdaL1, lambdaL2)
        
    
    def test(self, X_test : np.ndarray, Y_test : np.ndarray) -> np.ndarray :
        numpyLabels : np.ndarray = np.array(Y_test)
        uniqueLables = np.unique(numpyLabels)
        sortedLabels = np.sort(uniqueLables)

        predictionArray = []

        normalized_X_test = (X_test - self.train_mean) / self.train_std

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
                predictions = predictions[0][0]

            predictionArray.append([predictions, elemLabelsArray])

        accuracy /= X_test.shape[0]

        return accuracy, predictionArray
    
    def do_forwarding(self, input) -> None :
        layer : Layer = self.firstLayer
        layer.forwardPropagation(input)
        layer = layer.nextLayer

        while layer != None:
            layer.forwardPropagation()
            layer = layer.nextLayer

        return 
    
    def backpropagation_dataset(self, realValuesMatrix : np.ndarray, k : int) -> float :

        de_da_list : list[np.ndarray] = [None] * (self.hiddenLayerNum + 2)
        de_da_list[-1] = derivative_e_y(self.lastLayer.getOutput(), realValuesMatrix, self.isClassification) * 1
        
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

            ## Calcolo del prodotto esterno tra righe con stesso indice usando np.einsum
            row_wise_tensor : np.ndarray = np.einsum('ij,ik->ijk', de_da, da_dw)
            row_wise_outer : np.ndarray = row_wise_tensor.reshape((da_dw.shape[0], de_da.shape[1] * da_dw.shape[1]))

            dr_dw = row_wise_outer.sum(axis = 0)
            gradientSquaredNormArray.append(np.linalg.norm(dr_dw) ** 2)
            
            currLayer.update_weights(dr_dw, 0, k)

            currLayer = currLayer.nextLayer
            i += 1


        return np.sum(gradientSquaredNormArray)

    def backpropagation_sample(self, labels : np.ndarray, point_index : int) -> np.ndarray :
        layer : Layer = self.lastLayer

        de_da = []
        de_dw = []
        if (self.isClassification) :
            de_dy = derivative_e_y(layer.getOutput()[point_index][np.newaxis, :], labels, self.isClassification)
        else :
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

    def fit(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, epochs = 1e3, with_SAGA = False, show_error = False, with_replacement = False) -> tuple[list, list] :

        if (with_SAGA) :
            return self.__fit_saga(X_train, Y_train, epsilon, epochs, show_error, with_replacement)
        else :
            return self.__fit_dyn_sample(X_train, Y_train, epsilon, epochs, show_error, with_replacement)
    
    
    def __fit_dyn_sample(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, epochs = 1e3, show_error = False, with_replacement = False) -> tuple[list, list] :

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std

        if (self.isClassification) :
            numpyLabels : np.ndarray = np.array(Y_train)
            uniqueLables = np.unique(numpyLabels)
            sortedLabels = np.sort(uniqueLables)

            normalized_Y_train = Y_train
            sortedLabelsMatrix = np.tile(sortedLabels, (X_train.shape[0], 1))
            labelsMatrix = np.tile(Y_train.T, (sortedLabels.shape[0], 1)).T
            realValuesMatrix = (sortedLabelsMatrix == labelsMatrix).astype(int)
        else :
            self.train_y_mean = Y_train.mean(axis = 0)
            self.train_y_std = Y_train.std(axis = 0)
            normalized_Y_train = (Y_train - self.train_y_mean) / self.train_y_std
            realValuesMatrix = normalized_Y_train[:, np.newaxis]
            
        gradient_norm = epsilon
        gradient_norm_array = []
        error_array = []
        samples_seen = np.zeros(len(normalized_X_train))
        
        k = 1
        iter_num = 1
        total_iter_num = 1
        indexes = np.arange(0, len(normalized_X_train))
        min_err = float("inf")
        while (k <= epochs) :
            mini_batch_dim = min(int(1/32 * len(normalized_X_train) + iter_num), len(normalized_X_train))

            if (not with_replacement) : 
                ## Non prendo due volte lo stesso punto nella stessa epoca
                selectable_indexes = np.where(samples_seen == 0)[0]
                mini_batch_indexes = np.random.choice(a = indexes[selectable_indexes], size = min(mini_batch_dim, len(selectable_indexes)), replace = False)
            else :
                mini_batch_indexes = np.random.choice(a = len(normalized_X_train), size = mini_batch_dim, replace = False)
            mini_batch_train = normalized_X_train[mini_batch_indexes]

            self.do_forwarding(mini_batch_train)

            batchRealValuesMatrix = realValuesMatrix[mini_batch_indexes]
            gradientSquaredNorm = self.backpropagation_dataset(batchRealValuesMatrix, total_iter_num)
            gradient_norm = np.sqrt(gradientSquaredNorm)
            gradient_norm_array.append(gradient_norm)

            self.do_forwarding(normalized_X_train)

            if (self.isClassification) :
                error = middle_error(self.lastLayer.getOutput(), realValuesMatrix, self.isClassification)
            else :
                output = self.lastLayer.getOutput() * self.train_y_std + self.train_y_mean
                matrix = realValuesMatrix * self.train_y_std + self.train_y_mean
                error = middle_error(output, matrix, self.isClassification)
            print("Gradient's norm: ", gradient_norm, "--", "Error: ", error, "--", "Epoch: ", k)
            if (error < min_err) :
                min_err = error
                self.change_best_weights()
            error_array.append(error)

            iter_num += 1
            total_iter_num += 1
            samples_seen[mini_batch_indexes] = 1

            if (np.all(samples_seen == 1)) :
                k += 1
                iter_num = 1
                samples_seen[samples_seen == 1] = 0
        
        print("Errore minimo:", min_err)
        self.set_best_weights()
            
        return gradient_norm_array, error_array

    def __fit_saga(self, X_train : np.ndarray , Y_train : np.ndarray, epsilon = 1e-4, epochs = 1e3, show_error = False, with_replacement = False) -> tuple[list, list] :

        self.train_mean = X_train.mean(axis = 0)
        self.train_std = X_train.std(axis = 0) + 1e-3
        normalized_X_train = (X_train - self.train_mean) / self.train_std

        if (self.isClassification) :
            numpyLabels : np.ndarray = np.array(Y_train)
            uniqueLables = np.unique(numpyLabels)
            sortedLabels = np.sort(uniqueLables)

            sortedLabelsMatrix = np.tile(sortedLabels, (X_train.shape[0], 1))
            labelsMatrix = np.tile(Y_train.T, (sortedLabels.shape[0], 1)).T
            realValuesMatrix = (sortedLabelsMatrix == labelsMatrix).astype(int)
            normalized_Y_train = Y_train
            realValuesMatrix = (sortedLabelsMatrix == labelsMatrix).astype(int)
        else :
            self.train_y_mean = Y_train.mean(axis = 0)
            self.train_y_std = Y_train.std(axis = 0)
            normalized_Y_train = (Y_train - self.train_y_mean) / self.train_y_std
            realValuesMatrix = normalized_Y_train[:, np.newaxis]

        initialized_saga_acc = False
        gradient_norm = epsilon
        gradient_norm_array = []
        error_array = []
        samples_seen = np.zeros(len(normalized_X_train))
        
        k = 1
        iter_num = 1
        total_iter_num = 1
        indexes = np.arange(0, len(normalized_X_train))
        min_err = float("inf")
        while (gradient_norm >= epsilon and k <= epochs) :

            mini_batch_dim = min(int(1/32 * len(normalized_X_train) + iter_num), len(normalized_X_train))

            if (not with_replacement) : 
                ## Non prendo due volte lo stesso punto nella stessa epoca
                selectable_indexes = np.where(samples_seen == 0)[0]
                mini_batch_indexes = np.random.choice(a = indexes[selectable_indexes], size = min(mini_batch_dim, len(selectable_indexes)), replace = False)
            else :
                mini_batch_indexes = np.random.choice(a = len(normalized_X_train), size = mini_batch_dim, replace = False)
            mini_batch_train = normalized_X_train[mini_batch_indexes]

            self.do_forwarding(mini_batch_train)

            sagaIndex = np.random.randint(0, len(mini_batch_indexes))

            mini_batch_labels = normalized_Y_train[mini_batch_indexes]
            elemLabel = mini_batch_labels[sagaIndex]
            if (self.isClassification) :
                elemLabelsArray = (sortedLabels == elemLabel).astype(int)
            else :
                elemLabelsArray = elemLabel

            de_dw : np.ndarray = self.backpropagation_sample(elemLabelsArray, sagaIndex)
            
            if (not initialized_saga_acc) :
                accumulatorSAGA = np.zeros((X_train.shape[0], de_dw.shape[0]))
                initialized_saga_acc = True
            
            gradient_esteem = de_dw - (accumulatorSAGA[sagaIndex] - accumulatorSAGA.mean(axis = 0))
            
            accumulatorSAGA[sagaIndex] = de_dw

            gradient_norm = np.linalg.norm(gradient_esteem)
            gradient_norm_array.append(gradient_norm)

            self.do_forwarding(normalized_X_train)

            if (self.isClassification) :
                error = middle_error(self.lastLayer.getOutput(), realValuesMatrix, self.isClassification)
            else :
                output = self.lastLayer.getOutput() * self.train_y_std + self.train_y_mean
                matrix = realValuesMatrix * self.train_y_std + self.train_y_mean
                error = middle_error(output, matrix, self.isClassification)

            print("Gradient's norm: ", gradient_norm, "--", "Error: ", error, "--", "Epoch: ", k)
            if (error < min_err) :
                min_err = error
                self.change_best_weights()
            error_array.append(error)

            self.update_weights(gradient_esteem, total_iter_num)

            iter_num += 1
            total_iter_num += 1
            samples_seen[mini_batch_indexes] = 1

            if (np.all(samples_seen == 1)) :
                k += 1
                iter_num = 1
                samples_seen[samples_seen == 1] = 0
        
        print("Errore minimo:", min_err)
        self.set_best_weights()

        return gradient_norm_array, error_array


    def update_weights(self, gradient_esteem : np.ndarray, iter_num : int) -> None :
        layer = self.lastLayer
        i = 0
        start = 0
        while (layer.prevLayer != None) :
            layer.update_weights(gradient_esteem, start, iter_num)
            i += 1
            start = start + layer.neuronNumber * (layer.prevLayer.neuronNumber + 1)
            layer = layer.prevLayer
        return
    
    def change_best_weights(self) :
        currLayer : Layer = self.firstLayer.nextLayer
        while (currLayer != None) :
            currLayer.change_best_weights()
            currLayer = currLayer.nextLayer

    def set_best_weights(self) :
        currLayer : Layer = self.firstLayer.nextLayer
        while (currLayer != None) :
            currLayer.set_best_weights()
            currLayer = currLayer.nextLayer