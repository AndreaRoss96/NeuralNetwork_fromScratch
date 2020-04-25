import numpy as np
from NeuralNetwork import NeuralNetwork
import utils as utils
import time
from pandas import DataFrame

def main():
        csv_filename = "wine.csv"
        hidden_layers = [4] # number of nodes in hidden layers i.e. [layer1, layer2, ...]
        n_epochs = 30 # number of training epochs
        n_folds = 4 # number of folds for cross-validation
        crossval = 42 # seed for cross-validation
        functions = ["sigmoid", "sigmoid"] # list of activation functions
        batch_size = 5 # size of the batches

        X, y, n_classes = utils.read_csv(csv_filename, target="Class", norm=True)
        N, d = X.shape

        # Create cross-validation folds
        idx_all = np.arange(0, N)
        idx_folds = utils.crossval_folds(N, n_folds, seed=crossval) # list of list of fold indices

        # Train/evaluate the model on each fold
        acc_train, acc_valid = list(), list()
        for i, idx_valid in enumerate(idx_folds):
            # Collect training and test data from folds
            idx_train = np.delete(idx_all, idx_valid)
            X_train, y_train = X[idx_train], y[idx_train]
            X_valid, y_valid = X[idx_valid], y[idx_valid]

            model = NeuralNetwork(input_dim=d, hidden_layers=hidden_layers, output_dim=n_classes, functions=functions)
            model.fit(X_train, y_train, batch_size=batch_size, n_epochs=n_epochs)

            # Make predictions for training and test data

            ypred_train = model.predict(X_train)
            ypred_valid = model.predict(X_valid)

            # Compute training/test accuracy score from predicted values
            acc_train.append(100*np.sum(y_train==ypred_train)/len(y_train))
            acc_valid.append(100*np.sum(y_valid==ypred_valid)/len(y_valid))


            # Print cross-validation result
            print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}%".format(i+1, n_folds, acc_train[-1], acc_valid[-1]))

        # Print results
        print("  -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%".format(sum(acc_train)/float(len(acc_train)), sum(acc_valid)/float(len(acc_valid))))










if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    main()
