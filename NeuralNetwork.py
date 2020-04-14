import math, random
import numpy as np
from utils import dotprod
from functions import activation, derivative

class NeuralNetwork:

    def __init__(self, input_dim=None, hidden_layers=None, output_dim=None, functions=None, seed=1):
        '''
        The network is constructed so that it doesn't contain the input layer
        while its dimension is stored inside the class.
        The element of the input are passed as a parameter to the feedforward method
        '''
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None) or (len(functions) != len(hidden_layers)+1):
            raise Exception("Invalid arguments!")
        self.input_dim = input_dim # number of input nodes
        self.output_dim = output_dim # number of output nodes
        self.hidden_layers = hidden_layers # array with the number of node of each hidden layer
        self.functions = functions # list of activation functions
        self.net = self._build_network(seed=seed) # array of arrays of nodes
        self.mean_error = [ 0 for _ in range(len(self.net[-1]))] # array of zeros

    '''
    Train and fit the network
    With the default number of batch it performs a SGD,
    otherwise a bigger batch size means speed up and less precision
    '''
    def fit(self, input_val, output_val, l_rate=0.1, batch_size = 1, n_epochs=200):
        if batch_size != 0:
            batches = [input_val[x:x+batch_size] for x in range(0, len(input_val), batch_size)]
            true_outputs = [output_val[x:x+batch_size] for x in range(0, len(output_val), batch_size)]
        else:
            batches = [input_val]
            batch_size = len(input_val)
            true_outputs = [output_val]

        for _ in range(n_epochs):
            for (x_, y_) in zip(batches, true_outputs):
                self.mean_error = [ 0 for _ in range(len(self.net[-1]))]
                for i in range(len(x_)):
                    self._feedforward(x_[i]) # update node["a"] --> output of the network
                    out_ecoded = self._output_encoder(y_[i], self.output_dim)
                    for j, node in enumerate(self.net[-1]):
                        err = node["a"] - out_ecoded[j]
                        self.mean_error[j] += err
                for i in range(len(self.mean_error)):
                    self.mean_error[i] = self.mean_error[i] / batch_size
                self._backpropagation(out_ecoded) # update node["d"]
                self._update_weights(np.mean(x_, axis=0), l_rate) # update node["weight"]

    '''
    Predict the value choosing between the max value of of node['a'] in the output layer
    '''
    def predict(self, X):
        prediction = np.array([np.argmax(self._feedforward(x_)) for x_ in X], dtype=np.int)
        return prediction

    '''
    Build a dense netork layer by layer
    '''
    def _build_network(self, seed=1):
        random.seed(seed)
        # Create a single dense layer
        def _layer(input_dim, output_dim, layer_level):
            layer = []
            for _ in range(output_dim):
                weights = [random.random() for _ in range(input_dim)] # list of weights
                node = {
                    "weights": weights, # each node has the weights coming from the previous layer
                    "a": None,
                    "d": None,
                    "func" : self.functions[layer_level]
                    }
                layer.append(node)
            return layer

        # Stack layers (input -> hidden -> ... -> output)
        network = []
        if len(self.hidden_layers) == 0:
            # if there's no hidden layers, the functions build just the output layer
            network.append(_layer(self.input_dim, self.output_dim, 0))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0], 0))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i-1], self.hidden_layers[i], i))
            network.append(_layer(self.hidden_layers[-1], self.output_dim, len(self.hidden_layers)))
        return network

    def _feedforward(self, x):
        x_in = x
        for layer in self.net:
            x_out = []
            for node in layer:
                active = activation(node['func'])
                node["a"] = active(dotprod(node['weights'], x_in))
                x_out.append(node["a"])
            x_in = x_out # set output as next input
        return x_in

    def _backpropagation(self, out_ecoded):
        n_layers = len(self.net)
        for i in reversed(range(n_layers)):
            if i == n_layers - 1:
                for j, node in enumerate(self.net[i]):
                    active_derivative = derivative(node['func'])
                    node['d'] = self.mean_error[j] * active_derivative(node["a"])
            else:
                # Weighted sum of deltas from upper layer
                for j, node in enumerate(self.net[i]):
                    err = sum([upper_node['weights'][j] * upper_node['d'] for upper_node in self.net[i+1]])
                    active_derivative = derivative(node['func'])
                    node['d'] = err * active_derivative(node["a"])

    # updates node['weight']
    def _update_weights(self, x, l_rate):
        for i, layer in enumerate(self.net):
            if i == 0: inputs = x
            else: inputs = [node_["a"] for node_ in self.net[i-1]]
            # Update weights
            for node in layer:
                for j, input_ in enumerate(inputs):
                    # dw = - learning_rate * (error * active_func) * input
                    node['weights'][j] += - l_rate * node['d'] * input_

    def _output_encoder(self, idx, output_dim):
        x = np.zeros(output_dim, dtype=np.int)
        x[idx] = 1
        return x
