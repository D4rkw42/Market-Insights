// Define um neurônio LSTM na rede neural

#include "neural_network/Core/NeuralNetwork/Neurons/LSTMNeuron.hpp"

#include <effolkronium/random.hpp>

#include "neural_network/Math/ActivationFunctions/activationFunctions.hpp"

using random = effolkronium::random_static;

LSTMNeuron::LSTMNeuron(int weightsNum) : INeuron(weightsNum) {
    // Inicialização de pesos 

    for (int i = 0; i < weightsNum; ++i) {
        this->uf[i] = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);
        this->ui[i] = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);
        this->uc[i] = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);
    }

    this->wf = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);
    this->wi = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);
    this->wc = random::get<double>(-STANDARD_MAX_ABSOLUTE_WEIGHT, STANDARD_MAX_ABSOLUTE_WEIGHT);

    // Inicialização de bias

    this->bf = random::get<double>(-STANDAND_MAX_ABSOLUTE_BIAS, STANDAND_MAX_ABSOLUTE_BIAS);
    this->bi = random::get<double>(-STANDAND_MAX_ABSOLUTE_BIAS, STANDAND_MAX_ABSOLUTE_BIAS);
    this->bc = random::get<double>(-STANDAND_MAX_ABSOLUTE_BIAS, STANDAND_MAX_ABSOLUTE_BIAS);
}

//

double LSTMNeuron::Load(const std::vector<double>& input) {
    // Calculando a Forget Gate

    double ft = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        ft += this->uf[i] * input[i];
    }

    ft += this->wf * this->ht_1 + this->bf;
    ft = Sigmoid(ft);

    // Calculando a Input Gate

    double it = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        it += this->ui[i] * input[i];
    }

    it += this->wi * this->ht_1 + this->bi;
    it = Sigmoid(it);

    // Candidate Cell

    double candidate = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        candidate += this->uc[i] * input[i];
    }

    candidate += this->wc * this->ht_1 + this->bc;
    candidate = Tanh(candidate);

    // Cell State

    double ct = ft * this->ct_1 + it * candidate;
    this->ct_1 = ct;

    // Output Gate

    double ot = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        ot += this->weights[i] * input[i];
    }

    ot += this->wo * this->ht_1 + this->bias;
    ot = Sigmoid(ot);

    // Hidden State

    double ht = ot * Tanh(ct);
    this->ht_1 = ht;

    return ht;
}

void LSTMNeuron::UpdateWeightsAndBias(const std::vector<double>& lastInput, const double delta, const double learningRate) {

}
