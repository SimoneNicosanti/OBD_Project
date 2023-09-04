from NeuralNetwork import NeuralNetwork
from StepEnum import *
import numpy as np
from threading import *
import time
from LogWriter import *


def buildModel(
        isClassification : bool, 
        layerNumArray : list, 
        neuronNumArray : list,
        lambdaL1Array : list,
        lambdaL2Array : list,
        X_train,
        Y_train, 
        X_valid, 
        Y_valid, 
        epochs : int, 
        with_SAGA : bool, 
        method : StepEnum, 
        lambdaL1 : float,
        lambdaL2 : float,
        show_error : bool = False,
        crossValidation : bool = False,
        with_thread : bool = False,
        with_replacement : bool = False) -> NeuralNetwork :
    
    if (crossValidation and not with_thread) :
        ## Cross Validete Model
        return crossValidate(
            isClassification, 
            layerNumArray, 
            neuronNumArray, 
            lambdaL1Array,
            lambdaL2Array,
            X_train, Y_train, 
            X_valid, Y_valid,
            epochs,
            with_SAGA,
            method,
            lambdaL1,
            lambdaL2,
            show_error,
            with_replacement)
    elif (crossValidation and with_thread) :
        return crossValidate_thread(
            isClassification, 
            layerNumArray, 
            neuronNumArray, 
            lambdaL1Array,
            lambdaL2Array,
            X_train, Y_train, 
            X_valid, Y_valid,
            epochs,
            with_SAGA,
            method,
            lambdaL1,
            lambdaL2,
            show_error,
            with_replacement)
    else :
        ## Default Model
        featuresNumber, labelsNumber = getNetworkInputOutputDim(X_train, Y_train, isClassification)
        defaultNeuronNumber : int = getDefaultNeuronNumber(featuresNumber, labelsNumber)
        numberLayers = 2
        numberNeurons = [defaultNeuronNumber] * numberLayers

        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lambdaL1, lambdaL2)
        
        start = time.time()
        gradient_norm_array, error_array = model.fit(X_train, Y_train, epochs = epochs, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement)
        end = time.time()
        return model, gradient_norm_array, error_array, end - start



def crossValidate(
        isClassification : bool, 
        layerNumArray : list, 
        neuronNumArray : list,
        lambdaL1Array : list,
        lambdaL2Array : list,
        X_train, Y_train, 
        X_valid, Y_valid, 
        max_steps : int, 
        with_SAGA : bool, 
        method : StepEnum, 
        lambdaL1 : float,
        lambdaL2 : float,
        show_error : bool = False,
        with_replacement : bool = False) -> NeuralNetwork :

    featuresNumber, labelsNumber = getNetworkInputOutputDim(X_train, Y_train, isClassification)
    defaultNeuronNumber = getDefaultNeuronNumber(featuresNumber, labelsNumber)
    if (defaultNeuronNumber not in neuronNumArray) :
        neuronNumArray.append(defaultNeuronNumber)

    total_combinations : list = generate_combinations(neuronNumArray, layerNumArray)
 
    best_model : NeuralNetwork = None
    best_accuracy = 0.0
    best_error_array : list = None
    best_grad_norm_array : list = None
    best_time : float = None

    validStartTime = time.time()

    ## Cross Validation su regolarizzazione L1
    for i in range(0, len(total_combinations)) :
        numberNeurons = total_combinations[i]
        numberLayers = len(numberNeurons)

        for lamL1 in lambdaL1Array :
            if (lamL1 <= 0) :
                continue
            model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lamL1, 0)
            start = time.time()
            grad_norm_array, err_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement) 
            end = time.time()
            accuracy_validation, _ = model.test(X_valid, Y_valid)
            print("Model: ", numberNeurons, "--", "Accuracy validation: ", accuracy_validation, "--", "L1:", lamL1, "--", "L2", 0)

            if (accuracy_validation > best_accuracy) :
                best_model = model
                best_accuracy = accuracy_validation
                best_grad_norm_array = grad_norm_array
                best_error_array = err_array
                best_time = end - start
    
    ## Cross Validation su regolarizzazione L2
    for i in range(0, len(total_combinations)) :
        numberNeurons = total_combinations[i]
        numberLayers = len(numberNeurons)

        for lamL2 in lambdaL2Array :
            if (lamL2 <= 0) :
                continue
            model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, 0, lamL2)
            start = time.time()
            grad_norm_array, err_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement)
            end = time.time()
            accuracy_validation, _ = model.test(X_valid, Y_valid)
            print("Model: ", numberNeurons, "--", "Accuracy validation: ", accuracy_validation, "--", "L1:", 0, "--", "L2", lamL2)

            if (accuracy_validation > best_accuracy) :
                best_model = model
                best_accuracy = accuracy_validation
                best_grad_norm_array = grad_norm_array
                best_error_array = err_array
                best_time = end - start
    
    ## Cross Validation su valori di default
    for i in range(0, len(total_combinations)) :
        numberNeurons = total_combinations[i]
        numberLayers = len(numberNeurons)

        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification, method, lambdaL1, lambdaL2)
        start = time.time()
        grad_norm_array, err_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement)
        end = time.time()
        accuracy_validation, _ = model.test(X_valid, Y_valid)
        print("Model: ", numberNeurons, "--", "Accuracy validation: ", accuracy_validation, "--", "L1:", 0, "--", "L2", lamL2)

        if (accuracy_validation > best_accuracy) :
            best_model = model
            best_accuracy = accuracy_validation
            best_grad_norm_array = grad_norm_array
            best_error_array = err_array
            best_time = end - start 

    validEndTime = time.time()

    print("Time: ", validEndTime - validStartTime)

    print("Best accuracy: ", best_accuracy)

    return best_model, best_grad_norm_array, best_error_array, best_time


def getDefaultNeuronNumber(featuresNumber, labelsNumber) -> int :
    
    defaultNeuronNumber = int(2/3 * featuresNumber) + labelsNumber
    if (defaultNeuronNumber > 256) :
            defaultNeuronNumber = 128
    
    return defaultNeuronNumber

def getNetworkInputOutputDim(X_train, Y_train, isClassification : bool) :
    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1

    return featuresNumber, labelsNumber
    

def crossValidate_thread(
        isClassification : bool, 
        layerNumArray : list, 
        neuronNumArray : list,
        lambdaL1Array : list,
        lambdaL2Array : list,
        X_train, Y_train, 
        X_valid, Y_valid, 
        max_steps : int, 
        with_SAGA : bool, 
        method : StepEnum, 
        lambdaL1 : float,
        lambdaL2 : float,
        show_error : bool = False,
        with_replacement : bool = False) -> NeuralNetwork :

    featuresNumber, labelsNumber = getNetworkInputOutputDim(X_train, Y_train, isClassification)
    defaultNeuronNumber = getDefaultNeuronNumber(featuresNumber, labelsNumber)
    if (defaultNeuronNumber not in neuronNumArray) :
        neuronNumArray.append(defaultNeuronNumber)

    total_combinations : list = generate_combinations(neuronNumArray, layerNumArray)
 
    best_model : NeuralNetwork = None
    best_accuracy = 0.0
    best_error_array : list = None
    best_grad_norm_array : list = None
    best_time : float = None

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
        thread_array[thread_index] = Thread(target = thread_function, args = [X_train, Y_train, X_valid, Y_valid, featuresNumber, labelsNumber, lambdaL1Array, lambdaL2Array, isClassification, max_steps, with_SAGA, method, lambdaL1, lambdaL2, show_error, with_replacement, thread_index, thread_slice, thread_best_model_list])
        thread_array[thread_index].start()

    validateStartTime = time.time()
    for thread in thread_array :
        thread.join()
    validateEndTime = time.time()

    print("Time: ", validateEndTime - validateStartTime)

    for thread_index in range(0, thread_num) :
        thread_result = thread_best_model_list[thread_index]
        if (thread_result[0] > best_accuracy) :
            best_accuracy = thread_result[0]
            best_model = thread_result[1]
            best_grad_norm_array = thread_result[2]
            best_error_array = thread_result[3]
            best_time = thread_result[4]

    print("Best accuracy: ", best_accuracy)

    return best_model, best_grad_norm_array, best_error_array, best_time

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
        with_replacement : bool,
        thread_index : int, 
        thread_configuration_list : list, 
        thread_best_model_list : list
    ) -> None :

    best_model : NeuralNetwork = None
    best_accuracy = 0.0
    best_error_array : list = None
    best_grad_norm_array : list = None
    best_time : float = None
    
    for configuration in thread_configuration_list :

        for lamL1 in lambdaL1Array :
            if (lamL1 == 0) :
                continue
            model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification, method, lamL1, 0)
            start = time.time()
            grad_norm_array, err_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement) 
            end = time.time()
            accuracy_validation, _ = model.test(X_valid, Y_valid)
            print("Model: ", configuration, "--", "Accuracy validation: ", accuracy_validation, "--", "L1:", lamL1, "--", "L2", 0)

            if (accuracy_validation > best_accuracy) :
                best_model = model
                best_accuracy = accuracy_validation
                best_grad_norm_array = grad_norm_array
                best_error_array = err_array
                best_time = end - start
    
    ## Cross Validation su regolarizzazione L2
    for configuration in thread_configuration_list :
        for lamL2 in lambdaL2Array :
            if (lamL2 == 0) :
                continue
            model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification, method, 0, lamL2)
            start = time.time()
            grad_norm_array, err_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement)
            end = time.time()
            accuracy_validation, _ = model.test(X_valid, Y_valid)
            print("Model: ", configuration, "--", "Accuracy validation: ", accuracy_validation, "--", "L1:", 0, "--", "L2", lamL2)

            if (accuracy_validation > best_accuracy) :
                best_model = model
                best_accuracy = accuracy_validation
                best_grad_norm_array = grad_norm_array
                best_error_array = err_array
                best_time = end - start
    
    ## Cross Validation su valori di default
    for configuration in thread_configuration_list :

        model : NeuralNetwork = NeuralNetwork(len(configuration), featuresNumber, labelsNumber, configuration, isClassification, method, lambdaL1, lambdaL2)
        start = time.time()
        grad_norm_array, err_array = model.fit(X_train, Y_train, epochs = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA, show_error = show_error, with_replacement = with_replacement)
        end = time.time()
        accuracy_validation, _ = model.test(X_valid, Y_valid)
        print("Model: ", configuration, "--", "Accuracy validation: ", accuracy_validation, "--", "L1:", 0, "--", "L2", lamL2)

        if (accuracy_validation > best_accuracy) :
            best_model = model
            best_accuracy = accuracy_validation
            best_grad_norm_array = grad_norm_array
            best_error_array = err_array
            best_time = end - start 

    thread_best_model_list[thread_index] = (best_accuracy, best_model, best_grad_norm_array, best_error_array, best_time)