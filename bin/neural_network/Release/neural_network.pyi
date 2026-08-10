from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata', 'create_neural_network', 'cross_entropy', 'mean_absolute_error', 'mean_squared_error']
class NeuralNetwork:
    metadata: NeuralNetworkMetadata
    @staticmethod
    def load_neural_network(name: str) -> NeuralNetwork:
        ...
    @staticmethod
    def save_neural_network(neural_network: NeuralNetwork, name: str) -> bool:
        ...
    def back_propagation(self, expected: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], train_func_id: str, learning_rate: typing.SupportsFloat | typing.SupportsIndex) -> None:
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
def create_neural_network() -> NeuralNetwork:
    ...
def cross_entropy(output: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], expected: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    ...
def mean_absolute_error(output: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], expected: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    ...
def mean_squared_error(output: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], expected: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> float:
    ...
