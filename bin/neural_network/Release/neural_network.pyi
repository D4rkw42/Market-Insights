from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata']
class NeuralNetwork:
    metadata: NeuralNetworkMetadata
    def back_propagation(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def forward_pass(self, arg0: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex]) -> list[float]:
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
