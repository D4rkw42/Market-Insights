from __future__ import annotations
from neural_network.Release.neural_network import INeuronActivationFunction
from neural_network.Release.neural_network import INeuronActivationFunctionList
from neural_network.Release.neural_network import NeuralNetwork
from neural_network.Release.neural_network import NeuralNetworkArchitecture
from neural_network.Release.neural_network import NeuralNetworkArchitectureData
from neural_network.Release.neural_network import NeuralNetworkMetadata
from neural_network.Release.neural_network import TrainingErrorFunctionDx
from neural_network.Release.neural_network import TrainingErrorFunctionDxList
from neural_network.Release.neural_network import create_neural_network
from . import Release
__all__: list[str] = ['ACTIVATION_FUNCTION_LIST', 'INeuronActivationFunction', 'INeuronActivationFunctionList', 'NeuralNetwork', 'NeuralNetworkArchitecture', 'NeuralNetworkArchitectureData', 'NeuralNetworkMetadata', 'Release', 'TRAINING_ERROR_FUNCTION_DX_LIST', 'TrainingErrorFunctionDx', 'TrainingErrorFunctionDxList', 'create_neural_network']
ACTIVATION_FUNCTION_LIST: dict  # value = {'sigmoid': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD61430>, 'tanh': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD61450>, 'identity': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD61470>, 'leaky_relu100': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD61490>, 'leaky_relu1000': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD614B0>, 'gelu': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD614D0>}
TRAINING_ERROR_FUNCTION_DX_LIST: dict  # value = {'mse': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD614F0>, 'mae': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD61510>, 'cross_entropy': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000002D07CD61530>}
