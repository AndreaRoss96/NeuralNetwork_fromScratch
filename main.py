import numpy as np
from NeuralNetwork import NeuralNetwork
import utils as utils
import time
from pandas import DataFrame


def main():
        csv_filename = "C:\\Users\\orazi\\Desktop\\Filippo\\uni\\optimization\\progietto finale\\NuralNetworkGD\\wine.csv"
        hidden_layers = [4] # number of nodes in hidden layers i.e. [layer1, layer2, ...]
        eta = 0.1 # learning rate
        n_epochs = 400 # number of training epochs
        n_folds = 4 # number of folds for cross-validation
        seed_crossval = 1 # seed for cross-validation
        seed_weights = 1 # seed for NeuralNetwork weight initialization
        functions = ["sigmoid", "sigmoid"]

        # Read csv data + normalize features
        print("Reading '{}'...".format(csv_filename))
        X, y, n_classes = utils.read_csv(csv_filename, target_name="Class", normalize=True)
        print(" -> X.shape = {}, y.shape = {}, n_classes = {}\n".format(X.shape, y.shape, n_classes))
        N, d = X.shape

        print("Neural network model:")
        print(" input_dim = {}".format(d))
        print(" hidden_layers = {}".format(hidden_layers))
        print(" output_dim = {}".format(n_classes))
        print(" eta = {}".format(eta))
        print(" n_epochs = {}".format(n_epochs))
        print(" n_folds = {}".format(n_folds))
        print(" seed_crossval = {}".format(seed_crossval))
        print(" seed_weights = {}\n".format(seed_weights))

        # Create cross-validation folds
        idx_all = np.arange(0, N)
        idx_folds = utils.crossval_folds(N, n_folds, seed=seed_crossval) # list of list of fold indices

        # Train/evaluate the model on each fold
        acc_train, acc_valid = list(), list()
        print("Cross-validating with {} folds...".format(len(idx_folds)))
        for i, idx_valid in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_valid)
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]
        # Build neural network classifier model and train
        model = NeuralNetwork(input_dim=d, output_dim=n_classes, hidden_layers=hidden_layers, functions=functions, seed=seed_weights)
        model.fit(X_train, y_train, l_rate=eta, batch_size=5, n_epochs=n_epochs)

        # Make predictions for training and test data

        ypred_train = model.predict(X_train)
        #ypred_train = model.predict(df)
        ypred_valid = model.predict(X_valid)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(y_train==ypred_train)/len(y_train))
        acc_valid.append(100*np.sum(y_valid==ypred_valid)/len(y_valid))

        # Print cross-validation result
        print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
                i+1, n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))

        # Print results
        print("  -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%".format(
        sum(acc_train)/float(len(acc_train)), sum(acc_valid)/float(len(acc_valid))))


if __name__ == "__main__":
    main()
