// Define um neurônio LSTM na rede neural

#include "neural_network/Core/NeuralNetwork/Neurons/LSTMNeuron.hpp"

LSTMNeuron::LSTMNeuron(int weightsNum) : INeuron(weightsNum) {

}

//

double LSTMNeuron::Load(const std::vector<double>& input) const {

}

void LSTMNeuron::UpdateWeightsAndBias(const std::vector<double>& lastInput, const double delta, const double learningRate) {

}
