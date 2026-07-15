// Definição da interface de um neurônio na rede

#include "neural_network/Core/NeuralNetwork/INeuron.hpp"

INeuron::INeuron(int weightsNum) : weightsNum(weightsNum) {
    this->weights.resize(weightsNum);
    this->weights = { 0 };
}
