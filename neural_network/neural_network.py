import pandas as pd
import numpy as np
import math
from random import uniform

class NeuralNetwork(object):
    "Neural network with feed forward and back propagation"

    def __init__(self, topology, inputs, output, activation_scheme, weight=None):
        self.topology = topology
        self.layers = self.__setup_layers(topology, inputs)
        self.weights = self.__setup_weights(topology, weight)
        self.output = output
        self.activation_scheme = activation_scheme
        self.guess_list = list()
        self.error_list = list()
        self.layers_with_weights = list()

        def __relu_activation(x):
            return x if x > 0 else 0


        def __sigmoid_activation(x):
            return 1 / (1 + math.exp(-1 * x))

        self.activation_functions = {
            'Relu' : __relu_activation,
            'Sigmoid' : __sigmoid_activation
        }


    def __setup_layers(self, topology, inputs):
        topos = [None for x in topology]

        topos[0] = np.matrix(inputs)
        return topos


    def __setup_weights(self, topology, weight):
        weights = list()
        starting_weight = weight if weight is not None else uniform(0.00, 0.99)

        for layer_a, layer_b in zip(topology, topology[1:]):
            weights_arr = np.full((layer_a, layer_b), starting_weight)
            weights_matrix = np.matrix(weights_arr)
            weights.append(weights_matrix)

        return weights


    def feed_forward(self):

        def __compute_error_rate(guess, actual):
            return (guess - actual) ** 2

        for i in range(0, len(self.weights)):
            result_matrix = self.layers[i] * self.weights[i]

            key = self.activation_scheme[i]
            activation_function = np.vectorize(self.activation_functions[key])
            activated_matrix = activation_function(result_matrix)

            self.layers[i + 1] = activated_matrix

        #Append last layer as guess
        self.guess_list.append(self.layers[-1])

        #Compute error rate
        error_rate_func = np.vectorize(__compute_error_rate)
        error_rate = error_rate_func(self.guess_list[-1], self.output)
        self.error_list.append(error_rate)

        #Generate list of layers and weights
        layers_with_weights = [None] * (len(self.layers) + len(self.weights))
        layers_with_weights[::2] = self.layers
        layers_with_weights[1::2] = self.weights

        self.layers_with_weights.append(layers_with_weights)

        return layers_with_weights


    def back_propagation(self):

        def __compute_y_derivative(x):
            return (1 - x) * x

        def __compute_gradient(y, error):
            return y * error

        guess = self.guess_list[-1]
        error_rate = self.error_list[-1]
        prev_layer = self.layers[-2]
        prev_weight = self.weights[-1]

        y_derivative_func = np.vectorize(__compute_y_derivative)
        y_derivative = y_derivative_func(guess)

        gradient_func = np.vectorize(__compute_gradient)
        gradients = gradient_func(y_derivative, error_rate)
        gradients_tr = gradients.transpose()

        delta_w = gradients_tr * prev_layer
        delta_w_tr = delta_w.transpose()

        new_weight = prev_weight - delta_w_tr

        self.weights[-1] = new_weight


topology = [4, 2, 3]
inputs = [1, 0, 1, 0]
output = [1, 0, 0]
activation = ['Relu', 'Sigmoid']
neural_net = NeuralNetwork(topology, inputs, output, activation, 0.2)
topo = neural_net.feed_forward()
neural_net.back_propagation()
# print(str(topo))
print(str(neural_net.weights))
