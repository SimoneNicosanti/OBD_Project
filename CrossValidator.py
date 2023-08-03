from NeuralNetwork import NeuralNetwork
from StepEnum import *
import numpy as np
from threading import *
import time
from LogWriter import *


def crossValidate(
        isClassification : bool, 
        layerNumArray : list, 
        neuronNumArray : list,
        X_train,
        Y_train, 
        X_valid, 
        Y_valid, 
        max_steps : int, 
        with_SAGA : bool, 
        method : StepEnum, 
        crossValidation : bool = False) -> NeuralNetwork :
    

    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1

    magicNeuronNum : int = int(2/3 * featuresNumber) + labelsNumber
    if (magicNeuronNum not in neuronNumArray) :
        neuronNumArray.append(magicNeuronNum)

    # max_steps = 100
    # with_SAGA = True
    if (not crossValidation) :
        numberLayers = 2
        numberNeurons = [magicNeuronNum] * numberLayers
        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method)
        gradient_norm_array = model.fit(X_train, Y_train, max_steps = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA)
        writeAllNormLog(gradient_norm_array)
        return model

    total_combinations : list = generate_combinations(neuronNumArray, layerNumArray)
 
    best_accuracy = 0.0
    best_layers : int = 0
    best_neurons : list = []
    best_model : NeuralNetwork = None

    for i in range(0, len(total_combinations)) :
        numberNeurons = total_combinations[i]
        numberLayers = len(numberNeurons)
        print("Model: ", numberNeurons)
        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification)
        model.fit(X_train, Y_train, max_steps = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA) 
        accuracy_validation, _ = model.predict(X_valid, Y_valid)

        print("Model: ", numberNeurons, accuracy_validation)

        if (accuracy_validation >= best_accuracy) :
            best_accuracy = accuracy_validation
            best_layers = numberLayers
            best_neurons = numberNeurons
            best_model = model

    end = time.time()

    # print(end - start)

    print("Best accuracy: ", best_accuracy)
    # print("Best number of layers: ", best_layers)
    # print("Best combination of neurons: ", best_neurons)

    return best_model


def crossValidate_thread(isClassification : bool, layerNumArray : list, neuronNumArray : list, X_train, Y_train, X_valid, Y_valid, max_steps : int, with_SAGA : bool, method : StepEnum, crossValidation : bool = False) -> NeuralNetwork :

    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1

    magicNeuronNum : int = int(2/3 * featuresNumber) + labelsNumber
    if (magicNeuronNum not in neuronNumArray) :
        neuronNumArray.append(magicNeuronNum)

    # max_steps = 100
    # with_SAGA = True
    if (not crossValidation) :
        numberLayers = 2
        numberNeurons = [magicNeuronNum] * numberLayers
        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method)
        model.fit(X_train, Y_train, max_steps = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA)
        return model

 
    best_accuracy = 0.0
    best_layers : int = 0
    best_neurons : list = []
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
        thread_array[thread_index] = Thread(target = thread_function, args = [X_train, Y_train, X_valid, Y_valid, featuresNumber, labelsNumber, isClassification, max_steps, with_SAGA, thread_index, thread_slice, thread_best_model_list])
        thread_array[thread_index].start()

    start = time.time()
    for thread in thread_array :
        thread.join()
    end = time.time()

    print(end - start)

    for thread_index in range(0, thread_num) :
        thread_result = thread_best_model_list[thread_index]
        if (thread_result[0] > best_accuracy) :
            best_accuracy = thread_result[0]
            best_model = thread_result[1]


    print("Best accuracy: ", best_accuracy)
    # print("Best number of layers: ", best_layers)
    # print("Best combination of neurons: ", best_neurons)

    return best_model

# TODO : Regolarizzazione

def generate_combinations(neuronNumArray, layerNumArray : list) -> list:
    total_combinations = []
    for layerNum in layerNumArray:
        #layer_combinations = []
        generate_combinations_rec(neuronNumArray, layerNum, [], total_combinations)
        #total_combinations.append(layer_combinations)
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
        isClassification : bool,
        max_steps : int,
        with_SAGA : bool,
        thread_index : int, 
        thread_configuration_list : list, 
        thread_best_model_list : list
    ) -> None :

    best_accuracy = 0.0
    best_layers : int = 0
    best_neurons : list = []
    best_model : NeuralNetwork = None
    for configuration in thread_configuration_list :
        model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification)
        model.fit(X_train, Y_train, max_steps = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA) 
        accuracy_validation, _ = model.predict(X_valid, Y_valid)

        print(thread_index, configuration, accuracy_validation)

        if (accuracy_validation >= best_accuracy) :
            best_accuracy = accuracy_validation
            # best_layers = numberLayers
            # best_neurons = numberNeurons
            best_model = model

    thread_best_model_list[thread_index] = (best_accuracy, best_model)
