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

        self.activation_functions = self.__setup_activation_functions()
        self.derivative_functions = self.__setup_derivative_functions()


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


    def __setup_activation_functions(self):
        def __relu_activation(x):
            return x if x > 0 else 0

        def __sigmoid_activation(x):
            return 1 / (1 + math.exp(-1 * x))

        activation_functions = {
            'Relu' : __relu_activation,
            'Sigmoid' : __sigmoid_activation
        }

        return activation_functions


    def __setup_derivative_functions(self):
        def __relu_derivative(x):
            return 1 if x > 0 else 0

        def __sigmoid_derivative(x):
            return x * (1 - x)

        derivative_functions = {
            'Relu' : __relu_derivative,
            'Sigmoid' : __sigmoid_derivative
        }

        return derivative_functions


    def __generate_current_topology(self):
        "Generate list of layers and weights"

        layers_with_weights = [None] * (len(self.layers) + len(self.weights))
        layers_with_weights[::2] = self.layers
        layers_with_weights[1::2] = self.weights

        self.layers_with_weights = layers_with_weights


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

        self.__generate_current_topology()


    def back_propagation(self):

        def __compute_gradient(y, error):
            return y * error

        guess = self.guess_list[-1]
        error_rate = self.error_list[-1]

        #First part of back propagation
        prev_layer = self.layers[-2]
        prev_weight = self.weights[-1]

        key = self.activation_scheme[-1]
        y_derivative_func = np.vectorize(self.derivative_functions[key])
        y_derivative = y_derivative_func(guess)

        gradient_func = np.vectorize(__compute_gradient)
        gradients = gradient_func(y_derivative, error_rate)
        gradients_tr = gradients.transpose()

        delta_w = gradients_tr * prev_layer
        delta_w_tr = delta_w.transpose()

        new_weight = prev_weight - delta_w_tr

        self.weights[-1] = new_weight


        #Second Part of back propagation
        gradients_p = gradients
        weights_p = prev_weight

        for i in range(2, len(self.weights) + 1):
            layer = self.layers[-i]
            layer_next = self.layers[-(i+1)]

            key = self.activation_scheme[-i]
            derivative_function = np.vectorize(self.derivative_functions[key])
            z_hat = derivative_function(layer)

            weights_p_tr = weights_p.transpose()
            gradients_h = gradients_p * weights_p_tr
            gradients_h_activated = gradient_func(gradients_h, z_hat)

            layer_next_tr = layer_next.transpose()
            delta_w = layer_next_tr * gradients_h_activated

            original_weight = self.weights[-i]
            new_weight = original_weight - delta_w

            gradients_p = gradients_h_activated
            weights_p = original_weight

            self.weights[-i] = new_weight

        self.__generate_current_topology()


    def train(self, epochs):
        for i in range(0, epochs):
            print('Feed forward %i' % i)
            self.feed_forward()
            print(str(self.layers_with_weights))

            print('Back propagation %i' % i)
            self.back_propagation()
            print(str(self.layers_with_weights))            
