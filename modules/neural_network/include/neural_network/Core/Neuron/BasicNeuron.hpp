// Define um neurônio comum na rede neural

#pragma once

#include <neural_network/Core/Neuron/INeuron.hpp>

class BasicNeuron : public INeuron {
    public:
        BasicNeuron(int weightsNum);
        BasicNeuron(void) = default;

        virtual ~BasicNeuron() = default;

        double Load(const std::vector<double>& input) override;
        double Learn(double learningRate, double gradient) override;

        const std::vector<double> Weights(int ref) const override;

        const INeuronBuffer Serialize(void) const noexcept override; // Gera um buffer que representa todos os pesos e biases do neurônio
        void Deserialize(const double* buffer) noexcept override; // Carrega o buffer de pesos e biases do neurônio
};
