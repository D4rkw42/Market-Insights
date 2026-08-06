// Define um neurônio comum na rede neural

#pragma once

#include "neural_network/Core/NeuralNetwork/INeuron.hpp"

class BasicNeuron : public INeuron {
    public:
        BasicNeuron(int weightsNum);
        BasicNeuron(void) = default;

        virtual ~BasicNeuron() = default;

        double Load(const std::vector<double>& input) override;
        void UpdateWeightsAndBias(const std::vector<double>& lastInput, const double delta, const double learningRate) override;
};
