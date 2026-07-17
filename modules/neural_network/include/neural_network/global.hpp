// Definições globais

#pragma once

#include "neural_network/Core/NeuralNetwork/ActivationFunction.hpp"
#include "neural_network/Core/NeuralNetwork/ErrorFunction.hpp"

#include "neural_network/Math/ActivationFunctions/activationFunctions.hpp"
#include "neural_network/Math/ErrorFunctions/errorFunctions.hpp"

// Funções de ativação da biblioteca

inline ActivationFunctionList neuralNetworkActivationFunctions = {
    { "sigmoid", CreateActivationFunction(Sigmoid, SigmoidDx) },
    { "leaky_relu100", CreateActivationFunction(LeakyReLu100, LeakyReLu100Dx) },
    { "leaky_relu1000", CreateActivationFunction(LeakyReLu1000, LeakyReLu1000Dx) },
    { "gelu", CreateActivationFunction(GELU, GELUDx) },
    { "tanh", CreateActivationFunction(Tanh, TanhDx) },
    { "identity", CreateActivationFunction(Identity, IdentityDx) }
};

// Funções de erro da biblioteca

inline ErrorFunctionList neuralNetworkErrorFunctions = {
    { "mse", CreateErrorFunction(MeanSquaredError, MeanSquaredErrorDx) },
    { "mae", CreateErrorFunction(MeanAbsoluteError, MeanAbsoluteErrorDx) },
    { "cross-entropy", CreateErrorFunction(CrossEntropy, CrossEntropyDx) }
};
