// Define um neurônio comum na rede neural

#include "neural_network/Core/NeuralNetwork/Neurons/BasicNeuron.hpp"

BasicNeuron::BasicNeuron(int weightNum) : INeuron(weightNum) {}

//

double BasicNeuron::Load(const std::vector<double>& input) const {
    double z = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        z += input[i] * this->weights[i];
    }

    return z + this->bias;
}

void BasicNeuron::UpdateWeightsAndBias(const std::vector<double>& lastInput, const double delta, const double learningRate) {
    // Atualizando pesos
    for (int i = 0; i < this->weightsNum; ++i) {
        this->weights[i] -= learningRate * delta * lastInput[i];
    }

    // Atualizando bias
    this->bias -= learningRate * delta;
}
