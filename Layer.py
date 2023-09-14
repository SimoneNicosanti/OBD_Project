import numpy as np
from Utils import *

class Layer :

    def __init__(self, neuronNumber : int, lambdaL1 : float, lambdaL2 : float) -> None :
        self.neuronNumber = neuronNumber
        self.prevLayer = None
        self.nextLayer = None
        self.weightMatrix : np.ndarray = None
        self.biasArray : np.ndarray = np.zeros(neuronNumber)
        self.aArray : np.ndarray = np.zeros(neuronNumber)
        self.outputArray : np.ndarray = np.zeros(neuronNumber)
        self.lambdaL1 : float = lambdaL1
        self.lambdaL2 : float = lambdaL2

        self.bestWeightMatrix = None
        self.bestBiasArray = None

    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        self.prevLayer = prevLayer
        self.nextLayer = nextLayer

        if (prevLayer != None) :
            self.weightMatrix = np.random.randn(self.prevLayer.getNeuronNumber(), self.getNeuronNumber()) * np.sqrt(1 / (prevLayer.getNeuronNumber() + self.getNeuronNumber()))
            self.biasArray = np.zeros(self.getNeuronNumber())

    def getNeuronNumber(self) -> int :
        return self.neuronNumber

    def forwardPropagation(self, input : np.ndarray = None) -> None :
        if (self.prevLayer == None) :
            self.outputArray = input
        else :
            inputArray = self.prevLayer.getOutput()
            
            self.aArray = np.dot(inputArray, self.weightMatrix) - self.biasArray

            if (self.nextLayer == None) :
                self.outputArray = self.aArray
            else :
                self.outputArray = self.__relu(self.aArray)

        return

    def change_best_weights(self) :
        self.bestWeightMatrix = np.copy(self.weightMatrix)
        self.bestBiasArray = np.copy(self.biasArray)

    def set_best_weights(self) :
        self.weightMatrix = self.bestWeightMatrix
        self.biasArray = self.bestBiasArray

    def update_weights(self, gradient_esteem : np.ndarray, start : int, k : int, learning_rate : float = 0.001) -> None :
        end = start + (self.neuronNumber * (self.prevLayer.neuronNumber + 1))
        esteem_subset : np.ndarray = gradient_esteem[start : end]
        alpha = learning_rate / k
        esteem_matrix = esteem_subset.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T
        bias_incr = esteem_matrix[self.prevLayer.neuronNumber]
        weight_incr = esteem_matrix[0:self.prevLayer.neuronNumber]

        self.weightMatrix -= alpha * (weight_incr + self.lambdaL1 * np.sign(self.weightMatrix) + 2 * self.lambdaL2 * self.weightMatrix)
        self.biasArray -= alpha * (bias_incr + self.lambdaL1 * np.sign(self.biasArray) + 2 * self.lambdaL2 * self.biasArray)
        
    def getOutput(self) -> np.ndarray :
        return self.outputArray

    def __relu(self, z_array) -> np.ndarray :
        return np.maximum(z_array, 0)
    
    def relu_derivative(self, z_array : np.ndarray) :
        return (z_array > 0).astype(int)
    

class AdaGradLayer(Layer) :
    def __init__(self, neuronNumber : int, lambdaL1 : float, lambdaL2 : float) -> None :
        super().__init__(neuronNumber, lambdaL1, lambdaL2)
        self.adaGradAccumulator : np.ndarray = None
        
    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        super().setPrevAndNextLayer(prevLayer, nextLayer)

        if (prevLayer != None) :
            self.adaGradAccumulator = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))

    def update_weights(self, gradient_esteem : np.ndarray, start : int, k : int, learning_rate : float = 0.001) -> None :
        end = start + (self.neuronNumber * (self.prevLayer.neuronNumber + 1))
        esteem_subset : np.ndarray = gradient_esteem[start : end]
        esteem_matrix = esteem_subset.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T
        self.adaGradAccumulator += esteem_subset ** 2
        accum_matrix = self.adaGradAccumulator.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T

        alpha = learning_rate / (np.sqrt(accum_matrix) + 1e-8)
        bias_incr = esteem_matrix[self.prevLayer.neuronNumber]
        weight_incr = esteem_matrix[0:self.prevLayer.neuronNumber]
        bias_alpha = alpha[self.prevLayer.neuronNumber]
        weight_alpha = alpha[0:self.prevLayer.neuronNumber]

        self.biasArray -= bias_alpha * (bias_incr + self.lambdaL1 * np.sign(self.biasArray) + 2 * self.lambdaL2 * self.biasArray)
        self.weightMatrix -= weight_alpha * (weight_incr + self.lambdaL1 * np.sign(self.weightMatrix) + 2 * self.lambdaL2 * self.weightMatrix)


class RMSPropLayer(Layer) :
    def __init__(self, neuronNumber : int, lambdaL1 : float, lambdaL2 : float) -> None :
        super().__init__(neuronNumber, lambdaL1, lambdaL2)
        self.rmsPropAccumulator : np.ndarray = None

    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        super().setPrevAndNextLayer(prevLayer, nextLayer)

        if (prevLayer != None) :
            self.rmsPropAccumulator = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))

    def update_weights(self, gradient_esteem : np.ndarray, start : int, k : int, learning_rate : float = 0.001) -> None :
        end = start + (self.neuronNumber * (self.prevLayer.neuronNumber + 1))
        esteem_subset : np.ndarray = gradient_esteem[start : end]
        esteem_matrix = esteem_subset.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T
        self.rmsPropAccumulator = 0.9 * self.rmsPropAccumulator + 0.1 * esteem_subset ** 2
        rms_acc_mat = self.rmsPropAccumulator.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T
        
        alpha = learning_rate / (np.sqrt(rms_acc_mat) + 1e-5)
        bias_incr = esteem_matrix[self.prevLayer.neuronNumber]
        weight_incr = esteem_matrix[0:self.prevLayer.neuronNumber]
        bias_alpha = alpha[self.prevLayer.neuronNumber]
        weight_alpha = alpha[0:self.prevLayer.neuronNumber]

        self.weightMatrix -= weight_alpha * (weight_incr + self.lambdaL1 * np.sign(self.weightMatrix) + 2 * self.lambdaL2 * self.weightMatrix)
        self.biasArray -= bias_alpha * (bias_incr + self.lambdaL1 * np.sign(self.biasArray) + 2 * self.lambdaL2 * self.biasArray)



class AdamLayer(Layer) :
    def __init__(self, neuronNumber : int, lambdaL1 : float, lambdaL2 : float) -> None :
        super().__init__(neuronNumber, lambdaL1, lambdaL2)
        self.adamM : np.ndarray = None
        self.adamV : np.ndarray = None

    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        super().setPrevAndNextLayer(prevLayer, nextLayer)

        if (prevLayer != None) :
            self.adamM = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))
            self.adamV = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))

    def update_weights(self, gradient_esteem : np.ndarray, start : int, k : int, learning_rate : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999) -> None :
        end = start + (self.neuronNumber * (self.prevLayer.neuronNumber + 1))
        esteem_subset : np.ndarray = gradient_esteem[start : end]
        
        self.adamM = beta1 * self.adamM + (1 - beta1) * esteem_subset
        self.adamV = beta2 * self.adamV + (1 - beta2) * esteem_subset ** 2

        adam_m_matrix = self.adamM.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T
        adam_v_matrix = self.adamV.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T

        m_hat_mat = adam_m_matrix / (1 - beta1 ** k)
        v_hat_mat = adam_v_matrix / (1 - beta2 ** k)
        
        self.biasArray -= learning_rate * (m_hat_mat[self.prevLayer.neuronNumber] / (np.sqrt(v_hat_mat[self.prevLayer.neuronNumber]) + 1e-8) + self.lambdaL1 * np.sign(self.biasArray) + 2 * self.lambdaL2 * self.biasArray)
        self.weightMatrix -= learning_rate * (m_hat_mat[0 : self.prevLayer.neuronNumber] / (np.sqrt(v_hat_mat[0 : self.prevLayer.neuronNumber]) + 1e-8) + self.lambdaL1 * np.sign(self.weightMatrix) + 2 * self.lambdaL2 * self.weightMatrix)            


class NadamLayer(Layer) :
    def __init__(self, neuronNumber : int , lambdaL1 : float, lambdaL2 : float) -> None :
        super().__init__(neuronNumber, lambdaL1, lambdaL2)
        self.adamM : np.ndarray = None
        self.adamV : np.ndarray = None
        self.prevGradient : np.ndarray = None

    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        super().setPrevAndNextLayer(prevLayer, nextLayer)

        if (prevLayer != None) :
            self.adamM = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))
            self.adamV = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))
            self.prevGradient = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))

    def update_weights(self, gradient_esteem : np.ndarray, start : int, k : int, learning_rate : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999, gamma : float = 0.999) -> None :
        end = start + (self.neuronNumber * (self.prevLayer.neuronNumber + 1))
        esteem_subset : np.ndarray = gradient_esteem[start : end]
        esteem_matrix : np.ndarray = esteem_subset.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T

        self.adamM = beta1 * self.adamM + (1 - beta1) * esteem_subset
        self.adamV = beta2 * self.adamV + (1 - beta2) * esteem_subset ** 2

        adam_m_matrix = self.adamM.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T
        adam_v_matrix = self.adamV.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T

        m_hat_mat = adam_m_matrix / (1 - beta1 ** k)
        v_hat_mat = adam_v_matrix / (1 - beta2 ** k)
        nesterov = (1 - gamma) * esteem_matrix + gamma * self.prevGradient.reshape(self.neuronNumber, self.prevLayer.neuronNumber + 1).T

        self.weightMatrix -= learning_rate * ((m_hat_mat[0:self.prevLayer.neuronNumber] + gamma * nesterov[0:self.prevLayer.neuronNumber]) / (np.sqrt(v_hat_mat[0:self.prevLayer.neuronNumber]) + 1e-8) + self.lambdaL1 * np.sign(self.weightMatrix) + 2 * self.lambdaL2 * self.weightMatrix)
        self.biasArray -= learning_rate * ((m_hat_mat[self.prevLayer.neuronNumber] + gamma * nesterov[self.prevLayer.neuronNumber]) / (np.sqrt(v_hat_mat[self.prevLayer.neuronNumber]) + 1e-8) + self.lambdaL1 * np.sign(self.biasArray) + 2 * self.lambdaL2 * self.biasArray)

        self.prevGradient = esteem_subset


class AdaDeltaLayer(Layer) :
    def __init__(self, neuronNumber : int, lambdaL1 : float, lambdaL2 : float) -> None :
        super().__init__(neuronNumber, lambdaL1, lambdaL2)
        self.accumulatorG : np.ndarray = None
        self.accumulatorT : np.ndarray = None
        self.deltaT : np.ndarray = None

    def setPrevAndNextLayer(self, prevLayer, nextLayer) -> None :
        super().setPrevAndNextLayer(prevLayer, nextLayer)

        if (prevLayer != None) :
            self.accumulatorG = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))
            self.accumulatorT = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))
            self.deltaT = np.zeros(self.neuronNumber * (self.prevLayer.neuronNumber + 1))

    def update_weights(self, gradient_esteem : np.ndarray, start : int, k : int, rho : float = 0.001) -> None :
        end = start + (self.neuronNumber * (self.prevLayer.neuronNumber + 1))
        esteem_subset : np.ndarray = gradient_esteem[start : end]
        for j in range(0, self.neuronNumber) :
            for i in range(0, self.prevLayer.neuronNumber + 1) :
                index = j * (self.prevLayer.neuronNumber + 1) + i
                gradient_esteem_elem = esteem_subset[index]
                self.accumulatorG[index] = rho * self.accumulatorG[index] + (1 - rho) * gradient_esteem_elem ** 2
                self.deltaT[index] = - ((np.sqrt(np.diag(self.accumulatorT)[index][index] + 1e-8)) / (np.sqrt(np.diag(self.accumulatorG)[index][index] + 1e-8))) * gradient_esteem_elem
                self.accumulatorT = rho * self.accumulatorT + (1 - rho) * self.deltaT[index] ** 2
                if (i == self.prevLayer.neuronNumber) :
                    self.biasArray[j] += self.deltaT[index] + self.lambdaL1 * np.sign(self.biasArray[j]) + 2 * self.lambdaL2 * self.biasArray[j]
                else :
                    self.weightMatrix[i][j] += self.deltaT[index] + self.lambdaL1 * np.sign(self.weightMatrix[i][j]) + 2 * self.lambdaL2 * self.weightMatrix[i][j]

                