#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "neural_network/global.hpp"

#include "neural_network/Core/NeuralNetwork/NeuralNetwork.hpp"
#include "neural_network/Core/NeuralNetwork/ActivationFunction.hpp"
#include "neural_network/Core/NeuralNetwork/ErrorFunction.hpp"

namespace py = pybind11;

/* Definições gerais */

#define MODULE_NAME neural_network
#define MODULE_GIL_MODE py::mod_gil_not_used()

/* Definição da biblioteca */

PYBIND11_MODULE(MODULE_NAME, m, MODULE_GIL_MODE) {
    // Definições para função de erro e ativação

    py::class_<ActivationFunction, std::shared_ptr<ActivationFunction>>(m, "ActivationFunction");
    py::class_<ErrorFunction, std::shared_ptr<ErrorFunction>>(m, "ErrorFunction");

    py::class_<ActivationFunctionList>(m, "ActivationFunctionList");
    py::class_<ErrorFunctionList>(m, "ErrorFunctionList");

    m.attr("neural_network_activation_functions") = neuralNetworkActivationFunctions;
    m.attr("neural_network_error_functions") = neuralNetworkErrorFunctions;

    m.def("get_activation_function", &GetActivationFunction, py::arg("list"), py::arg("function"));
    m.def("get_error_function", &GetErrorFunction, py::arg("list"), py::arg("function"));

    // Metadados da rede neural (tipagem

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
        .def("back_propagation", &NeuralNetwork::BackPropagation, py::arg("expected"), py::arg("error_function"));
}
