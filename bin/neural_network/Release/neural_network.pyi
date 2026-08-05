from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['ActivationFunction', 'ActivationFunctionList', 'ErrorFunction', 'ErrorFunctionList', 'NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata', 'get_activation_function', 'get_error_function', 'neural_network_activation_functions', 'neural_network_error_functions']
class ActivationFunction:
    pass
class ActivationFunctionList:
    pass
class ErrorFunction:
    pass
class ErrorFunctionList:
    pass
class NeuralNetwork:
    metadata: NeuralNetworkMetadata
    def back_propagation(self, expected: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], error_function: ErrorFunction) -> None:
        ...
    def forward_pass(self, inputs: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> list[float]:
        ...
class NeuralNetworkArchitecture:
    pass
class NeuralNetworkArchitectureData:
    activation_function: str
    type: str
    @property
    def neurons(self) -> int:
        ...
    @neurons.setter
    def neurons(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class NeuralNetworkMetadata:
    created_at: str
    name: str
    @property
    def architecture(self) -> list[NeuralNetworkArchitectureData]:
        ...
    @architecture.setter
    def architecture(self, arg0: collections.abc.Sequence[NeuralNetworkArchitectureData]) -> None:
        ...
    @property
    def fitness(self) -> float:
        ...
    @fitness.setter
    def fitness(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def pairs_trained(self) -> list[str]:
        ...
    @pairs_trained.setter
    def pairs_trained(self, arg0: collections.abc.Sequence[str]) -> None:
        ...
def get_activation_function(list: collections.abc.Mapping[str, ActivationFunction], function: str) -> ActivationFunction:
    ...
def get_error_function(list: collections.abc.Mapping[str, ErrorFunction], function: str) -> ErrorFunction:
    ...
neural_network_activation_functions: dict  # value = {'sigmoid': <neural_network.Release.neural_network.ActivationFunction object>, 'leaky_relu100': <neural_network.Release.neural_network.ActivationFunction object>, 'leaky_relu1000': <neural_network.Release.neural_network.ActivationFunction object>, 'gelu': <neural_network.Release.neural_network.ActivationFunction object>, 'tanh': <neural_network.Release.neural_network.ActivationFunction object>, 'identity': <neural_network.Release.neural_network.ActivationFunction object>}
neural_network_error_functions: dict  # value = {'mse': <neural_network.Release.neural_network.ErrorFunction object>, 'mae': <neural_network.Release.neural_network.ErrorFunction object>, 'cross-entropy': <neural_network.Release.neural_network.ErrorFunction object>}
