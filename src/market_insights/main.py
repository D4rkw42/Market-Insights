# An application built upon a LSTM neural network that creates insights about financial market for buying or selling operations
# D4rkw42

from neural_network import *

def main():
    neural_network = NeuralNetwork.load_neural_network("test01")

    result = neural_network.forward_pass([0.5, 0.1])
    expected = [0.5, 0.2]

    err1 = mean_squared_error(result, expected)
    err2 = mean_absolute_error(result, expected)
    err3 = cross_entropy(result, expected)

    print(result)
    print(err1, err2, err3)
