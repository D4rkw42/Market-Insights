#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "neural_network/Core/NeuralNetwork/NeuralNetwork.hpp"

namespace py = pybind11;

/* Definições gerais */

#define MODULE_NAME neural_network
#define MODULE_GIL_MODE py::mod_gil_not_used()

/* Definição da biblioteca */

PYBIND11_MODULE(MODULE_NAME, m, MODULE_GIL_MODE) {
    // Metadados da rede neural (tipagem)
    
    py::class_<NeuralNetworkArchitectureData>(m, "NeuralNetworkArchitectureData")
        .def_readwrite("neurons", &NeuralNetworkArchitectureData::neurons)
        .def_readwrite("type", &NeuralNetworkArchitectureData::type)
        .def_readwrite("activation_function", &NeuralNetworkArchitectureData::activationFunction);

    py::class_<NeuralNetworkArchitecture>(m, "NeuralNetworkArchitecture");

    py::class_<NeuralNetworkMetadata>(m, "NeuralNetworkMetadata")
        .def_readwrite("name", &NeuralNetworkMetadata::name)
        .def_readwrite("created_at", &NeuralNetworkMetadata::createdAt)
        .def_readwrite("fitness", &NeuralNetworkMetadata::fitness)
        .def_readwrite("pairs_trained", &NeuralNetworkMetadata::pairsTrained)
        .def_readwrite("architecture", &NeuralNetworkMetadata::architecture);

    // Classe básica de rede neural
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def_readwrite("metadata", &NeuralNetwork::metadata)
        .def("forward_pass", &NeuralNetwork::ForwardPass)
        .def("back_propagation", &NeuralNetwork::BackPropagation);
}
