// Definições globais

#pragma once

#include <functional>
#include <string>
#include <map>

#include <neural_network/Math/ActivationFunctions/activationFunctions.hpp>
#include <neural_network/Math/ErrorFunctions/errorFunctions.hpp>

// Definição de uma função de erro da rede
using TrainingErrorFunction = std::function<double(const std::vector<double>&, const std::vector<double>&)>;
using TrainingErrorFunctionDx = std::function<double(double, double)>;

// Lista de funções de erro da rede
using TrainingErrorFunctionList = std::map<std::string, const TrainingErrorFunction>;
using TrainingErrorFunctionDxList = std::map<std::string, const TrainingErrorFunctionDx>;

// Definição de função de ativação de um neurônio e sua derivada
using INeuronActivationFunction = std::function<double(double)>;
using INeuronActivationFunctionDx = INeuronActivationFunction;

// Definição da lista de funções de ativação do neurônio
using INeuronActivationFunctionList = std::map<std::string, const INeuronActivationFunction>;
using INeuronActivationFunctionDxList = std::map<std::string, const INeuronActivationFunctionDx>;

// Lista de funções de ativação e suas derivadas

inline INeuronActivationFunctionList ACTIVATION_FUNCTION_LIST = {
    { "sigmoid", Sigmoid }, { "tanh", Tanh }, { "identity", Identity },
    { "leaky_relu100", LeakyReLu1000 }, { "leaky_relu1000", LeakyReLu1000 }, { "gelu", GELU }
};

inline INeuronActivationFunctionDxList ACTIVATION_FUNCTION_DX_LIST = {
    { "sigmoid", SigmoidDx }, { "tanh", TanhDx }, { "identity", IdentityDx },
    { "leaky_relu100", LeakyReLu1000Dx }, { "leaky_relu1000", LeakyReLu1000Dx }, { "gelu", GELUDx }
};

// Lista de funções de erro e suas derivadas

inline TrainingErrorFunctionList TRAINING_ERROR_FUNCTION_LIST = {
    { "mse", MeanSquaredError },
    { "mae", MeanAbsoluteError },
    { "cross_entropy", CrossEntropy }
};

inline TrainingErrorFunctionDxList TRAINING_ERROR_FUNCTION_DX_LIST = {
    { "mse", MeanSquaredErrorDx },
    { "mae", MeanAbsoluteErrorDx },
    { "cross_entropy", CrossEntropyDx }
};
