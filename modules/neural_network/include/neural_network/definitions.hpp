// Definições globais

#pragma once

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>
#include <neural_network/Core/Neuron/INeuron.hpp>

#include <neural_network/Math/ActivationFunctions/activationFunctions.hpp>
#include <neural_network/Math/ErrorFunctions/errorFunctions.hpp>

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
