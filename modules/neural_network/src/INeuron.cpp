// Definição da interface de um neurônio na rede

#include "neural_network/Core/NeuralNetwork/INeuron.hpp"

#include <effolkronium/random.hpp>

using random = effolkronium::random_static;

INeuron::INeuron(int weightsNum) : weightsNum(weightsNum) {
    this->weights.resize(weightsNum);

    // Bias inicial aleatório
    this->bias = random::get<double>(-STANDAND_MAX_ABSOLUTE_BIAS, STANDAND_MAX_ABSOLUTE_BIAS);

    // Pesos iniciais aleatórios
    for (int i = 0; i < weightsNum; ++i) {
        this->weights[i] = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);
    }
}
