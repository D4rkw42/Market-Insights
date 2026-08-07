#include <memory>

#include <pybind11/pybind11.h>

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>

#include <pybind11/operators.h>
#include <pybind11/iostream.h>

#include <pybind11/numpy.h>

#include <neural_network/definitions.hpp>

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>
#include <neural_network/Core/Neuron/INeuron.hpp>

namespace py = pybind11;

/* Definições gerais */

#define MODULE_NAME neural_network
#define MODULE_GIL_MODE py::mod_gil_not_used()

/* Definição da biblioteca */

PYBIND11_MODULE(MODULE_NAME, m, MODULE_GIL_MODE) {
    // Definições globais

    py::class_<INeuronActivationFunction>(m, "INeuronActivationFunction");
    py::class_<TrainingErrorFunctionDx>(m, "TrainingErrorFunctionDx");

    py::class_<INeuronActivationFunctionList>(m, "INeuronActivationFunctionList");
    py::class_<TrainingErrorFunctionDxList>(m, "TrainingErrorFunctionDxList");

    m.attr("ACTIVATION_FUNCTION_LIST") = ACTIVATION_FUNCTION_LIST;
    m.attr("TRAINING_ERROR_FUNCTION_DX_LIST") = TRAINING_ERROR_FUNCTION_DX_LIST;

    // Metadados da rede neural (tipagem)

    py::class_<NeuralNetworkArchitectureData, std::shared_ptr<NeuralNetworkArchitectureData>>(m, "NeuralNetworkArchitectureData")
        .def_readwrite("neurons", &NeuralNetworkArchitectureData::neurons)
        .def_readwrite("type", &NeuralNetworkArchitectureData::type)
        .def_readwrite("activation_function", &NeuralNetworkArchitectureData::activationFunction);

    py::class_<NeuralNetworkArchitecture>(m, "NeuralNetworkArchitecture");

    py::class_<NeuralNetworkMetadata, std::shared_ptr<NeuralNetworkMetadata>>(m, "NeuralNetworkMetadata")
        .def_readwrite("name", &NeuralNetworkMetadata::name)
        .def_readwrite("created_at", &NeuralNetworkMetadata::createdAt)
        .def_readwrite("fitness", &NeuralNetworkMetadata::fitness)
        .def_readwrite("pairs_trained", &NeuralNetworkMetadata::pairsTrained)
        .def_readwrite("architecture", &NeuralNetworkMetadata::architecture);

    // Classe básica de rede neural
    py::class_<NeuralNetwork, std::shared_ptr<NeuralNetwork>>(m, "NeuralNetwork")
        .def_readwrite("metadata", &NeuralNetwork::metadata)
        .def("forward_pass", &NeuralNetwork::ForwardPass, py::arg("inputs"))
        .def("back_propagation", &NeuralNetwork::BackPropagation, py::arg("expected"), py::arg("training_error_function"))
        .def_static("save_neural_network", &NeuralNetwork::SaveNeuralNetwork, py::arg("neural_network"), py::arg("name"))
        .def_static("load_neural_network", &NeuralNetwork::LoadNeuralNetwork, py::arg("name"));

    // Criação dinâmica de uma rede neural
    m.def("create_neural_network", &CreateNeuralNetwork);
}
