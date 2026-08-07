from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['ACTIVATION_FUNCTION_LIST', 'INeuronActivationFunction', 'INeuronActivationFunctionList', 'NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata', 'TRAINING_ERROR_FUNCTION_DX_LIST', 'TrainingErrorFunctionDx', 'TrainingErrorFunctionDxList']
class INeuronActivationFunction:
    pass
class INeuronActivationFunctionList:
    pass
class NeuralNetwork:
    metadata: NeuralNetworkMetadata
    def back_propagation(self, expected: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], training_error_function: collections.abc.Callable[[typing.SupportsFloat | typing.SupportsIndex, typing.SupportsFloat | typing.SupportsIndex], float]) -> None:
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
ACTIVATION_FUNCTION_LIST: dict  # value = {'sigmoid': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F9219705B0>, 'tanh': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F9219705D0>, 'identity': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F9219705F0>, 'leaky_relu100': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F921970610>, 'leaky_relu1000': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F921970630>, 'gelu': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F921970650>}
TRAINING_ERROR_FUNCTION_DX_LIST: dict  # value = {'mse': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F921970670>, 'mae': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F921970690>, 'cross_entropy': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001F9219706B0>}
