from NeuralNetwork import NeuralNetwork
from StepEnum import *
import numpy as np
from threading import *
import time
from LogWriter import *

# TODO : verificare cross validation
def crossValidate(
        isClassification : bool, 
        layerNumArray : list, 
        neuronNumArray : list,
        lambdaL1Array : list,
        lambdaL2Array : list,
        X_train,
        Y_train, 
        X_valid, 
        Y_valid, 
        max_steps : int, 
        with_SAGA : bool, 
        method : StepEnum, 
        lambdaL1 : float,
        lambdaL2 : float,
        show_error : bool = False,
        crossValidation : bool = False) -> NeuralNetwork :

    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1

    defaultNeuronNumber : int = int(2/3 * featuresNumber) + labelsNumber
    if (defaultNeuronNumber > 256) :
        defaultNeuronNumber = 128
    if (defaultNeuronNumber not in neuronNumArray) :
        neuronNumArray.append(defaultNeuronNumber)

    if (not crossValidation) :
        numberLayers = 5
        numberNeurons = [defaultNeuronNumber] * numberLayers
        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lambdaL1, lambdaL2)
        gradient_norm_array, error_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error)
        return model, gradient_norm_array, error_array

    total_combinations : list = generate_combinations(neuronNumArray, layerNumArray)
 
    best_accuracy = 0.0
    best_model : NeuralNetwork = None

    start = time.time()

    for i in range(0, len(total_combinations)) :
        numberNeurons = total_combinations[i]
        numberLayers = len(numberNeurons)
        if (len(lambdaL1Array) == 0) :
            for j in range(0, len(lambdaL2Array)) :
                model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, 0, lambdaL2Array[j])
                model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error) 
                accuracy_validation, _ = model.predict(X_valid, Y_valid)
                print("Model: ", numberNeurons, "--", "Accuracy validation: ", accuracy_validation, "--", "lambdaL2: ", lambdaL2Array[j])

        elif (len(lambdaL2Array) == 0) :
            for j in range(0, len(lambdaL1Array)) :
                model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lambdaL1Array[j], 0)
                model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error) 
                accuracy_validation, _ = model.predict(X_valid, Y_valid)
                print("Model: ", numberNeurons, "--", "Accuracy validation: ", accuracy_validation, "--", "lambdaL1: ", lambdaL1Array[j])
        else :
            model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lambdaL1, lambdaL2)
            model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error) 
            accuracy_validation, _ = model.predict(X_valid, Y_valid)
            print("Model: ", numberNeurons, "--", "Accuracy validation: ", accuracy_validation)

        if (accuracy_validation >= best_accuracy) :
            best_accuracy = accuracy_validation
            best_model = model

    end = time.time()

    print("Time: ", end - start)

    print("Best accuracy: ", best_accuracy)

    return best_model


def crossValidate_thread(
        isClassification : bool, 
        layerNumArray : list, 
        neuronNumArray : list, 
        lambdaL1Array : list,
        lambdaL2Array : list,
        X_train, Y_train, 
        X_valid, Y_valid, 
        epochs : int, 
        with_SAGA : bool, 
        method : StepEnum, 
        lambdaL1 : float, 
        lambdaL2 : float, 
        show_error : bool = False,
        crossValidation : bool = False
        ) -> NeuralNetwork :

    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1

    magicNeuronNum : int = int(2/3 * featuresNumber) + labelsNumber
    if (magicNeuronNum not in neuronNumArray) :
        neuronNumArray.append(magicNeuronNum)

    if (not crossValidation) :
        numberLayers = 3
        numberNeurons = [100] * numberLayers
        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lambdaL1, lambdaL2)
        model.fit(X_train, Y_train, epochs = epochs, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error)
        return model
 
    best_accuracy = 0.0
    best_model : NeuralNetwork = None

    total_combinations : list = generate_combinations(neuronNumArray, layerNumArray)
    thread_num = 4
    thread_best_model_list : list = [None] * thread_num

    thread_slices = []
    for i in range(0, thread_num) :
        thread_slices.append([])
    for i in range(0, len(total_combinations)) :
        thread_slices[i % thread_num].append(total_combinations[i])

    thread_array : list[Thread] = [None] * thread_num
    for thread_index in range(0, thread_num) :
        thread_slice = thread_slices[thread_index]
        thread_array[thread_index] = Thread(target = thread_function, args = [X_train, Y_train, X_valid, Y_valid, featuresNumber, labelsNumber, lambdaL1Array, lambdaL2Array, isClassification, epochs, with_SAGA, method, lambdaL1, lambdaL2, show_error, thread_index, thread_slice, thread_best_model_list])
        thread_array[thread_index].start()

    start = time.time()
    for thread in thread_array :
        thread.join()
    end = time.time()

    print("Time: ", end - start)

    for thread_index in range(0, thread_num) :
        thread_result = thread_best_model_list[thread_index]
        if (thread_result[0] > best_accuracy) :
            best_accuracy = thread_result[0]
            best_model = thread_result[1]

    print("Best accuracy: ", best_accuracy)

    return best_model

def generate_combinations(neuronNumArray, layerNumArray : list) -> list:
    total_combinations = []
    for layerNum in layerNumArray:
        generate_combinations_rec(neuronNumArray, layerNum, [], total_combinations)
    return total_combinations

def generate_combinations_rec(neuronNumArray, current_layers, current_combination, layer_combinations):
        if current_layers == 0:
            layer_combinations.append(current_combination)
            return

        for neurons in neuronNumArray:
            generate_combinations_rec(neuronNumArray, current_layers - 1, current_combination + [neurons], layer_combinations)

def thread_function(
        X_train : np.ndarray, 
        Y_train : np.ndarray,
        X_valid : np.ndarray,
        Y_valid : np.ndarray,
        featuresNumber : int,
        labelsNumber : int,
        lambdaL1Array : list,
        lambdaL2Array : list,
        isClassification : bool,
        max_steps : int,
        with_SAGA : bool,
        method : StepEnum,
        lambdaL1 : float,
        lambdaL2 : float,
        show_error : bool,
        thread_index : int, 
        thread_configuration_list : list, 
        thread_best_model_list : list
    ) -> None :

    best_accuracy = 0.0
    best_model : NeuralNetwork = None
    for configuration in thread_configuration_list :
        if (len(lambdaL1Array) == 0) :
            for j in range(0, len(lambdaL2Array)) :
                model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification, method, 0, lambdaL2Array[j])
                model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error) 
                accuracy_validation, _ = model.predict(X_valid, Y_valid)
                print("Thread: ", thread_index, "Model: ", configuration, "--", "Accuracy validation: ", accuracy_validation, "--", "lambdaL2: ", lambdaL2Array[j])

        if (len(lambdaL2Array) == 0) :
            for j in range(0, len(lambdaL1Array)) :
                model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification, method, lambdaL1Array[j], 0)
                model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error) 
                accuracy_validation, _ = model.predict(X_valid, Y_valid)
                print("Thread: ", thread_index, "Model: ", configuration, "--", "Accuracy validation: ", accuracy_validation, "--", "lambdaL1: ", lambdaL1Array[j])

        else :
            model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification, method, lambdaL1, lambdaL2)
            model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error) 
            accuracy_validation, _ = model.predict(X_valid, Y_valid)
            print("Thread: ", thread_index, "Model: ", configuration, "--", "Accuracy validation: ", accuracy_validation)

        if (accuracy_validation >= best_accuracy) :
            best_accuracy = accuracy_validation
            best_model = model

    thread_best_model_list[thread_index] = (best_accuracy, best_model)
