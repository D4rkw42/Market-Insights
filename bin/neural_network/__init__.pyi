from __future__ import annotations
from neural_network.Release.neural_network import ActivationFunction
from neural_network.Release.neural_network import ActivationFunctionList
from neural_network.Release.neural_network import ErrorFunction
from neural_network.Release.neural_network import ErrorFunctionList
from neural_network.Release.neural_network import NeuralNetwork
from neural_network.Release.neural_network import NeuralNetworkArchitecture
from neural_network.Release.neural_network import NeuralNetworkArchitectureData
from neural_network.Release.neural_network import NeuralNetworkMetadata
from neural_network.Release.neural_network import get_activation_function
from neural_network.Release.neural_network import get_error_function
from . import Release
__all__: list[str] = ['ActivationFunction', 'ActivationFunctionList', 'ErrorFunction', 'ErrorFunctionList', 'NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata', 'Release', 'get_activation_function', 'get_error_function', 'neural_network_activation_functions', 'neural_network_error_functions']
neural_network_activation_functions: dict  # value = {'sigmoid': <neural_network.Release.neural_network.ActivationFunction object>, 'leaky_relu100': <neural_network.Release.neural_network.ActivationFunction object>, 'leaky_relu1000': <neural_network.Release.neural_network.ActivationFunction object>, 'gelu': <neural_network.Release.neural_network.ActivationFunction object>, 'tanh': <neural_network.Release.neural_network.ActivationFunction object>, 'identity': <neural_network.Release.neural_network.ActivationFunction object>}
neural_network_error_functions: dict  # value = {'mse': <neural_network.Release.neural_network.ErrorFunction object>, 'mae': <neural_network.Release.neural_network.ErrorFunction object>, 'cross-entropy': <neural_network.Release.neural_network.ErrorFunction object>}
