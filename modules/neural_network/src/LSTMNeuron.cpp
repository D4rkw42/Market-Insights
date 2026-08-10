// Define um neurônio LSTM na rede neural

#include <neural_network/Core/Neuron/LSTMNeuron.hpp>

#include <effolkronium/random.hpp>

#include <neural_network/Math/ActivationFunctions/activationFunctions.hpp>

using random = effolkronium::random_static;

LSTMNeuron::LSTMNeuron(int weightsNum) : INeuron(weightsNum) {
    // Inicialização de pesos 

    this->uf.resize(weightsNum);
    this->ui.resize(weightsNum);
    this->uc.resize(weightsNum);

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

    for (int i = 0; i < this->weightsNum; ++i) {
        this->ft += this->uf[i] * input[i];
    }

    this->ft += this->wf * this->ht_1 + this->bf;
    this->ft = Sigmoid(this->ft);

    // Calculando a Input Gate

    for (int i = 0; i < this->weightsNum; ++i) {
        this->it += this->ui[i] * input[i];
    }

    this->it += this->wi * this->ht_1 + this->bi;
    this->it = Sigmoid(this->it);

    // Candidate Cell

    this->candidate = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        this->candidate += this->uc[i] * input[i];
    }

    this->candidate += this->wc * this->ht_1 + this->bc;
    this->candidate = Tanh(this->candidate);

    // Cell State

    double ct = this->ft * this->ct_1 + this->it * this->candidate;
    this->ct_1 = ct;

    // Output Gate

    for (int i = 0; i < this->weightsNum; ++i) {
        this->ot += this->weights[i] * input[i];
    }

    this->ot += this->wo * this->ht_1 + this->bias;
    this->ot = Sigmoid(this->ot);

    // Hidden State

    double ht = this->ot * this->actFunc.Activate(ct);
    this->ht_1 = ht;

    // Atualizando informações
    this->x = input;
    this->z = ot;
    this->a = ht;

    return ht;
}

double LSTMNeuron::Learn(double learningRate, double gradient) {
    // Calculando todos os gradientes parciais

    double outputGr = gradient * this->actFunc.Activate(this->ct_1);
    double cellStGr = gradient * this->ot * this->actFunc.Derivative(this->ct_1);

    double inputGr = cellStGr * this->candidate;
    double candidateGr = cellStGr * this->it;

    double forgetGr = cellStGr * this->ct_1;

    double prevCellStGr = cellStGr * this->ft;

    // Atualizando todos os pesos e bias

    // Forget Gate

    for (int i = 0; i < this->weightsNum; ++i) {
        this->uf[i] -= learningRate * forgetGr * this->ft * (1 - this->ft) * this->x[i];
    }

    this->wf -= learningRate * forgetGr * this->ft * (1 - this->ft) * this->ht_1;
    this->bf -= learningRate * forgetGr * this->ft * (1 - this->ft);

    // Input Gate

    for (int i = 0; i < this->weightsNum; ++i) {
        this->ui[i] -= learningRate * inputGr * this->it * (1 - this->it) * this->x[i];
    }

    this->wi -= learningRate * inputGr * this->it * (1 - this->it) * this->ht_1;
    this->bi -= learningRate * inputGr * this->it * (1 - this->it);

    // Candidate

    for (int i = 0; i < this->weightsNum; ++i) {
        this->uc[i] -= learningRate * candidateGr * this->candidate * (1 - this->candidate) * this->x[i];
    }

    this->wc -= learningRate * candidateGr * this->candidate * (1 - this->candidate) * this->ht_1;
    this->bc -= learningRate * candidateGr * this->candidate * (1 - this->candidate);

    // Output Gate

    for (int i = 0; i < this->weightsNum; ++i) {
        this->weights[i] -= learningRate * outputGr * this->ot * (1 - this->ot) * this->x[i];
    }

    this->wo -= learningRate * outputGr * this->ot * (1 - this->ot) * this->ht_1;
    this->bias -= learningRate * outputGr * this->ot * (1 - this->ot);

    //

    double delta = forgetGr + inputGr + candidateGr + outputGr;

    return delta;
}

const std::vector<double> LSTMNeuron::Weights(int ref) const {
    return std::vector<double> { this->uf[ref], this->ui[ref], this->uc[ref], this->weights[ref] };
}

//

const INeuronBuffer LSTMNeuron::Serialize(void) const noexcept {
    std::size_t elements = this->weightsNum * 4 + 8;

    double* buffer = new double[elements];
    std::size_t size = elements * sizeof(double);
    
    int ref = 0;
    int i;

    for (i = 0; i < this->weightsNum; ++i) {
        int id = i + ref;
        buffer[id] = this->uf[i];
    }

    ref += this->weightsNum;

    for (i = 0; i < this->weightsNum; ++i) {
        int id = i + ref;
        buffer[id] = this->ui[i];
    }

    ref += this->weightsNum;

    for (i = 0; i < this->weightsNum; ++i) {
        int id = i + ref;
        buffer[id] = this->uc[i];
    }

    ref += this->weightsNum;

    for (i = 0; i < this->weightsNum; ++i) {
        int id = i + ref;
        buffer[id] = this->weights[i];
    }

    ref += this->weightsNum;

    buffer[ref++] = this->wf;
    buffer[ref++] = this->wi;
    buffer[ref++] = this->wc;
    buffer[ref++] = this->wo;

    buffer[ref++] = this->bf;
    buffer[ref++] = this->bi;
    buffer[ref++] = this->bc;
    buffer[ref] = this->bias;

    return INeuronBuffer { buffer, size };
}

void LSTMNeuron::Deserialize(const double* buffer) noexcept {
    int ref = 0;

    // Pesos para as entradas

    for (int i = 0; i < this->weightsNum; ++i) {
        this->uf[i] = buffer[i + ref];
    }

    ref += this->weightsNum;

    for (int i = 0; i < this->weightsNum; ++i) {
        this->ui[i] = buffer[i + ref];
    }

    ref += this->weightsNum;

    for (int i = 0; i < this->weightsNum; ++i) {
        this->uc[i] = buffer[i + ref];
    }

    ref += this->weightsNum;

    for (int i = 0; i < this->weightsNum; ++i) {
        this->weights[i] = buffer[i + ref];
    }

    ref += this->weightsNum;

    // Pesos para o estado oculto

    this->wf = buffer[ref++];
    this->wi = buffer[ref++];
    this->wc = buffer[ref++];
    this->wo = buffer[ref++];

    // Biases

    this->bf = buffer[ref++];
    this->bi = buffer[ref++];
    this->bc = buffer[ref++];
    this->bias = buffer[ref];
}
