// Define um neurônio LSTM na rede neural

#pragma once

#include "neural_network/Core/NeuralNetwork/INeuron.hpp"

class LSTMNeuron : public INeuron {
    LSTMNeuron(int weightsNum);
    LSTMNeuron(void) = default;

    virtual ~LSTMNeuron() = default;

    double Load(const std::vector<double>& input) const override;
    void UpdateWeightsAndBias(const std::vector<double>& lastInput, const double delta, const double learningRate) override;
};
