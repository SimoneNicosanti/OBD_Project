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