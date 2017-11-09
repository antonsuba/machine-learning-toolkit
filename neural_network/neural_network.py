import pandas as pd
import numpy as np
import math
from random import uniform

class NeuralNetwork(object):
    "Neural network with feed forward and back propagation"

    def __init__(self, topology, inputs, output, activation_scheme, weight=None):
        self.topology = topology
        self.layers = self.__setup_topology(inputs)
        self.weights = self.__setup_weights(topology, weight)
        self.activation_scheme = activation_scheme

        def __relu_activation(x):
            return x if x > 0 else 0


        def __sigmoid_activation(x):
            return 1 / (1 + math.exp(-1 * x))

        self.activation_functions = {
            'Relu' : __relu_activation,
            'Sigmoid' : __sigmoid_activation
        }


    def __setup_topology(self, inputs):
        topos = list()
        topos.append(np.matrix(inputs))
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
        for i in range(0, len(self.weights)):
            result_matrix = self.layers[i] * self.weights[i]

            # print(str(self.layers))
            key = self.activation_scheme[i]
            activation_function = np.vectorize(self.activation_functions[key])
            activated_matrix = activation_function(result_matrix)

            self.layers.append(activated_matrix)


topology = [4, 2, 3]
inputs = [1, 0, 1, 0]
output = [1, 0, 0]
activation = ['Relu', 'Sigmoid']
neural_net = NeuralNetwork(topology, inputs, output, activation, 0.2)
neural_net.feed_forward()
print(str(neural_net.layers))
