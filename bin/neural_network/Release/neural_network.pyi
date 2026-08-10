from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['ACTIVATION_FUNCTION_LIST', 'INeuronActivationFunction', 'INeuronActivationFunctionList', 'NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata', 'TRAINING_ERROR_FUNCTION_DX_LIST', 'TrainingErrorFunctionDx', 'TrainingErrorFunctionDxList', 'create_neural_network']
class INeuronActivationFunction:
    pass
class INeuronActivationFunctionList:
    pass
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
class TrainingErrorFunctionDx:
    pass
class TrainingErrorFunctionDxList:
    pass
def create_neural_network() -> NeuralNetwork:
    ...
ACTIVATION_FUNCTION_LIST: dict  # value = {'gelu': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A96841430>, 'identity': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A96841450>, 'leaky_relu100': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A96841470>, 'leaky_relu1000': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A96841490>, 'sigmoid': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A968414B0>, 'tanh': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A968414D0>}
TRAINING_ERROR_FUNCTION_DX_LIST: dict  # value = {'cross_entropy': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A968414F0>, 'mae': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A96841510>, 'mse': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x0000023A96841530>}
