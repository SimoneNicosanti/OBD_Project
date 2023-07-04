import pandas as pd
from Utils import train_test_split
from NeuralNetwork import NeuralNetwork

CSV_URL = "https://raw.githubusercontent.com/ProfAI/tutorials/master/Come%20Creare%20una%20Rete%20Neurale%20da%20Zero/breast_cancer.csv"

breast_cancer = pd.read_csv(CSV_URL)
X = breast_cancer.drop("malignant", axis=1).values
y = breast_cancer["malignant"].values

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3)

X_max = X_train.max(axis=0)
X_min = X_train.min(axis=0)

X_train = (X_train - X_min)/(X_max-X_min)
X_test = (X_test - X_min)/(X_max-X_min)

model = NeuralNetwork(5, 30, 2, 100)
#model.fit(X_train, y_train, epochs=500, lr=0.01)

print(model.predict(X_test))