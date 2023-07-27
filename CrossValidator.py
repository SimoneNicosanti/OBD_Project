from NeuralNetwork import NeuralNetwork
import numpy as np

# TODO : Cross Validation con thread
def crossValidate(isClassification : bool, layerNumArray : list, neuronNumArray : list, X_train, Y_train, X_valid, Y_valid, crossValidation : bool = False) -> NeuralNetwork :

    featuresNumber = X_train.shape[1]
    if (isClassification) :
        labelsNumber = len(np.unique(Y_train))
    else :
        labelsNumber = 1

    magicNeuronNum : int = int(2/3 * featuresNumber) + labelsNumber
    if (magicNeuronNum not in neuronNumArray) :
        neuronNumArray.append(magicNeuronNum)

    if (not crossValidation) :
        numberLayers = 2
        numberNeurons = [magicNeuronNum] * numberLayers
        model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification)
        return model

    max_steps = 100
    with_SAGA = True
    total_combinations : list = generate_combinations(neuronNumArray, layerNumArray)
    best_accuracy = 0.0
    best_layers : int = 0
    best_neurons : list = []
    best_model : NeuralNetwork = None

    for i in range(0, len(layerNumArray)) :
        for j in range(0, len(total_combinations[i])) :
            numberLayers = layerNumArray[i]
            numberNeurons = total_combinations[i][j]
            model : NeuralNetwork = NeuralNetwork(numberLayers, featuresNumber, labelsNumber, numberNeurons, isClassification)
            model.fit(X_train, Y_train, max_steps = max_steps, epsilon = 1e-12, with_SAGA = with_SAGA) 
            accuracy_validation, _ = model.predict(X_valid, Y_valid) 

            print("Model: ", numberLayers, numberNeurons, accuracy_validation)

            if (accuracy_validation >= best_accuracy) :
                best_accuracy = accuracy_validation
                best_layers = numberLayers
                best_neurons = numberNeurons
                best_model = model

    print("Best accuracy: ", best_accuracy)
    print("Best number of layers: ", best_layers)
    print("Best combination of neurons: ", best_neurons)

    return best_model

# TODO : Regolarizzazione

def generate_combinations(neuronNumArray, layerNumArray : list) -> list:
    total_combinations = []
    for layerNum in layerNumArray:
        layer_combinations = []
        generate_combinations_rec(neuronNumArray, layerNum, [], layer_combinations)
        total_combinations.append(layer_combinations)
    return total_combinations

def generate_combinations_rec(neuronNumArray, current_layers, current_combination, layer_combinations):
        if current_layers == 0:
            layer_combinations.append(current_combination)
            return

        for neurons in neuronNumArray:
            generate_combinations_rec(neuronNumArray, current_layers - 1, current_combination + [neurons], layer_combinations)