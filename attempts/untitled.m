classdef NeuralNetwork
    properties
        hidden_layer_size
        w1
        b1
        W2
        b2
        forward_cache
    end
    
    methods
        function obj = NeuralNetwork(hidden_layer_size)
            if nargin < 1
                hidden_layer_size = 100;
            end
            obj.hidden_layer_size = hidden_layer_size;
        end
        
        function obj = init_weights(obj, input_size, hidden_size)
            obj.w1 = randn(input_size, hidden_size);
            obj.b1 = zeros(hidden_size, 1);
            obj.W2 = randn(hidden_size, 1);
            obj.b2 = 0;
        end
        
        function accuracy = accuracy(obj, y, y_pred)
            accuracy = sum(y == y_pred) / length(y);
        end
        
        function log_loss = log_loss(obj, y_true, y_proba)
            log_loss = -sum(y_true .* log(y_proba) + (1 - y_true) .* log(1 - y_proba)) / length(y_true);
        end
        
        function A = relu(obj, Z)
            A = max(Z, 0);
        end
        
        function A = sigmoid(obj, Z)
            A = 1 ./ (1 + exp(-Z));
        end
        
        function dZ = relu_derivative(obj, Z)
            dZ = zeros(size(Z));
            dZ(Z > 0) = 1;
        end
        
        function proba = forward_propagation(obj, X)
            Z1 = X * obj.w1 + obj.b1';
            A1 = obj.relu(Z1);
            Z2 = A1 * obj.W2 + obj.b2;
            A2 = obj.sigmoid(Z2);
            obj.forward_cache = {Z1, A1, Z2, A2};
            proba = A2(:);
        end
        
        function [y, proba] = predict(obj, X, return_proba)
            if nargin < 3
                return_proba = false;
            end
            proba = obj.forward_propagation(X);
            y = zeros(size(X, 1), 1);
            y(proba >= 0.5) = 1;
            y(proba < 0.5) = 0;
            if return_proba
                y = {y, proba};
            end
        end
        
        function [dW1, db1, dW2, db2] = back_propagation(obj, X, y)
            [Z1, A1, Z2, A2] = obj.forward_cache{:};
            m = size(A1, 1);
            dZ2 = A2 - reshape(y, [], 1);
            dW2 = A1' * dZ2 / m;
            db2 = sum(dZ2) / m;
            dZ1 = (dZ2 * obj.W2') .* obj.relu_derivative(Z1);
            dW1 = X' * dZ1 / m;
            db1 = sum(dZ1) / m;
        end
        
        function obj = fit(obj, X, y, epochs, lr)
            if nargin < 5
                lr = 0.01;
            end
            if nargin < 4
                epochs = 200;
            end
            obj.init_weights(size(X, 2), obj.hidden_layer_size);
            for i = 1:epochs
                Y = obj.forward_propagation(X);
                [dW1, db1, dW2, db2] = obj.back_propagation(X, y);
                obj.w1 = obj.w1 - lr * dW1;
                obj.b1 = obj.b1 - lr * db1';
                obj.W2 = obj.W2 - lr * dW2;
                obj.b2 = obj.b2 - lr * db2;
            end
        end
        
        function [accuracy, log_loss] = evaluate(obj, X, y)
            [y_pred, proba] = obj.predict(X, true);
            accuracy = obj.accuracy(y, y_pred);
            log_loss = obj.log_loss(y, proba);
        end
    end
end

function [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size, random_state)
    if nargin < 4
        random_state = [];
    end
    if nargin < 3
        test_size = 0.3;
    end
    if ~isempty(random_state)
        rng(random_state);
    end
    n = size(X, 1);
    test_indices = randperm(n, floor(n * test_size));
    X_test = X(test_indices, :);
    y_test = y(test_indices);
    X_train = X(setdiff(1:n, test_indices), :);
    y_train = y(setdiff(1:n, test_indices));
end

CSV_URL = 'https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv';
breast_cancer = readtable(CSV_URL);
X = table2array(breast_cancer(:, 1:end-1));
y = table2array(breast_cancer(:, end));
[X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.3);
X_max = max(X_train);
X_min = min(X_train);
X_train = (X_train - X_min) ./ (X_max - X_min);
X_test = (X_test - X_min) ./ (X_max - X_min);
model = NeuralNetwork();
model.fit(X_train, y_train, 500, 0.01);
[accuracy, log_loss] = model.evaluate(X_test, y_test);
disp([accuracy, log_loss]);
