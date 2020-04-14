import pandas as pd
import numpy as np
from neural_network import NeuralNetwork
from NodeNeuralNetwork import NodeNeuralNetwork
from sklearn.model_selection import train_test_split
import random
from src.NN import NN


def main():
    csv_filename = "/home/andreaross/Desktop/provaNNscratch/NN-scratch/wine.csv"
    hidden_layers = [5] # number of nodes in hidden layers i.e. [layer1, layer2, ...]
    eta = 0.1 # learning rate
    n_epochs = 400 # number of training epochs
    n_folds = 4 # number of folds for cross-validation
    seed_crossval = 1 # seed for cross-validation
    seed_weights = 1 # seed for NN weight initialization
    n_classes

    ds = pd.read_csv("wine.data")
    seed = 42
    #seed = random.randint(0,100)
    Y = ds["Class"]
    X = ds.drop("Class", axis = 1)
    i_train, i_test, o_train, o_test = train_test_split(
            X, Y, train_size=0.99, random_state = seed)
    print(i_train.shape)
    model = NN(input_dim=d, output_dim=n_classes, hidden_layers=hidden_layers, seed=seed_weights)
    model.train(X_train, y_train, eta=eta, n_epochs=n_epochs)
    nn.fit(i_train, o_train, epoch=50, batch_size=1)
    # print(nn.errors)

if __name__ == "__main__":
    main()
