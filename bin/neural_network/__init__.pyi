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
ACTIVATION_FUNCTION_LIST: dict  # value = {'gelu': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F0805B0>, 'identity': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F0805D0>, 'leaky_relu100': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F0805F0>, 'leaky_relu1000': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F080610>, 'sigmoid': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F080630>, 'tanh': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F080650>}
TRAINING_ERROR_FUNCTION_DX_LIST: dict  # value = {'cross_entropy': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F080670>, 'mae': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F080690>, 'mse': <built-in method  of pybind11_builtins.pybind11_detail_function_record_v1_msvc_md_mscver19 object at 0x000001D40F0806B0>}
