# An application built upon a LSTM neural network that creates insights about financial market for buying or selling operations
# D4rkw42

from neural_network import NeuralNetwork, create_neural_network

def main():
    neural_network = NeuralNetwork.load_neural_network("test01")

    values = [1, 1]

    result = neural_network.forward_pass(values)

    print(result)

    neural_network.back_propagation([0, 0.11], "mse", 0.02)

    result = neural_network.forward_pass(values)

    print(result)
