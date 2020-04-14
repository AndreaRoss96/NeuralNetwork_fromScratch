import numpy as np
'''
Sigmoid is used in binary classification problem [-1,1]
ReLu is mainly used to perform threshold operation to each input elem where val <0 is set to 0 (better performance in generalization)
Softmax is mainly used for multi-class problem
'''
#def sigmoid = lambda x: 1 / (1 + np.exp(-x))
def sigmoid(x): return (1/(1 + np.exp(-x)))

def relu(x): return np.maximum(0,x)

def hyperbolic(x): return np.tanh(x)

def softmax(x):
    # return an np.array
    exp = np.exp(x)
    exp_sum = np.sum(np.exp(x))
    return exp/exp_sum



'''
v DERIVATIVE v
'''
def sigmoid_deriv(x): return (sigmoid(x)*(1-sigmoid(x)))

def relu_deriv(x):
  if x >= 0: return 1
  else: return 0

def hyperbolic_deriv(x): return (1/np.cosh(x))**2

def softmax_deriv(x):
    # example x[0] = i  x[-1] = l
    #(e^i*sum(e^j)) / (sum(e^n))^2)   -- j:[i+1, l] and n:[i, l]
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis=0)

'''
v COST FUNCTIONS v
'''
def mean_squared_error(x, y):
    # works for np's arrays
    return np.square(x - y).mean()

def cross_entropy(m, a, Y):
    # m: number of samples
    # a: activation values in the output layer
    # Y: true values
    return -(1/m) * np.sum(Y*np.log(a) + (1-Y)*np.log(1-a))

'''
v COST FUNCTIONS DERIVATIVE v
'''

def mean_squared_error_der(x, y):
    return 2*( x - y )


'''
Functions selctors

With the string passed as a value it is possible to get the desired function
'''
def error_function_getter(func):
    options = {
        "mean_squared_error": mean_squared_error,
        "cross_entropy":cross_entropy,
    }
    return options.get(func.lower())

def error_f_deriv_getter(func):
    options = {
        "mean_squared_error": mean_squared_error_der,
        "cross_entropy":mean_squared_error_der,
    }
    return options.get(func.lower())

def activation(func):
    options = {
        "sigmoid": sigmoid,
        "relu":relu,
        "hyperbolic":hyperbolic,
        "softmax":softmax
    }
    return options.get(func.lower())

def derivative(func):
    options = {
        "sigmoid": sigmoid_deriv,
        "relu":relu_deriv,
        "hyperbolic":hyperbolic_deriv,
        "softmax":softmax_deriv
    }
    return options.get(func.lower())
