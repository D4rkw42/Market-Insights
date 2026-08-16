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

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>
#include <neural_network/Core/NeuralNetwork/NeuralNetworkSchema.hpp>
#include <neural_network/Core/Neuron/INeuron.hpp>

#include <neural_network/Math/ActivationFunctions/activationFunctions.hpp>
#include <neural_network/Math/ErrorFunctions/errorFunctions.hpp>

namespace py = pybind11;

/* Definições gerais */

#define MODULE_NAME neural_network
#define MODULE_GIL_MODE py::mod_gil_not_used()

/* Definição da biblioteca */

PYBIND11_MODULE(MODULE_NAME, m, MODULE_GIL_MODE) {
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
        .def(py::init<>())
        .def_readwrite("metadata", &NeuralNetwork::metadata)
        .def("forward_pass", &NeuralNetwork::ForwardPass, py::arg("inputs"))
        .def("back_propagation", &NeuralNetwork::BackPropagation, py::arg("expected"), py::arg("train_func_id"), py::arg("learning_rate") = DEFAULT_LEARNING_RATE)
        .def_static("save_neural_network", py::overload_cast<const std::shared_ptr<NeuralNetwork>&, const std::string&>(&NeuralNetwork::SaveNeuralNetwork), py::arg("neural_network"), py::arg("name"))
        .def_static("save_neural_network", py::overload_cast<const std::shared_ptr<NeuralNetwork>&>(&NeuralNetwork::SaveNeuralNetwork), py::arg("neural_network"))
        .def_static("load_neural_network", &NeuralNetwork::LoadNeuralNetwork, py::arg("name"));

    // Exportação das funções de erro

    m.def("mean_squared_error", &MeanSquaredError, py::arg("output"), py::arg("expected"));
    m.def("mean_absolute_error", &MeanAbsoluteError, py::arg("output"), py::arg("expected"));
    m.def("cross_entropy", &CrossEntropy, py::arg("output"), py::arg("expected"));

    // Exportando função SoftMax e sua derivada

    m.def("softmax", &SoftMax, py::arg("logits"));
    m.def("softmax_dx", &SoftMaxDx, py::arg("softmax"), py::arg("expected"));

    // Criação dinâmica de redes neurais

    py::class_<NeuralNetworkSchema, std::shared_ptr<NeuralNetworkSchema>>(m, "NeuralNetworkSchema")
        .def(py::init<>())
        .def("create_layer_schema_at", &NeuralNetworkSchema::CreateLayerSchemaAt, py::arg("neurons"), py::arg("type"), py::arg("activation_function"), py::arg("id"))
        .def("create_layer_schema", &NeuralNetworkSchema::CreateLayerSchema, py::arg("neurons"), py::arg("type"), py::arg("activation_function"))
        .def("remove_layer_schema_at", &NeuralNetworkSchema::RemoveLayerSchemaAt, py::arg("id"))
        .def("remove_layer_schema", &NeuralNetworkSchema::RemoveLayerSchema)
        .def("generate_neural_network", &NeuralNetworkSchema::GenerateNeuralNetwork, py::arg("name"))
        .def("get_amount_of_schemas", &NeuralNetworkSchema::GetAmountOfSchemas);
}
