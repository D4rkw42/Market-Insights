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

    double ft = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        ft += this->uf[i] * input[i];
    }

    ft += this->wf * this->ht + this->bf;

    double zft = ft;
    ft = Sigmoid(zft);

    // Calculando a Input Gate

    double it = 0;
    
    for (int i = 0; i < this->weightsNum; ++i) {
        it += this->ui[i] * input[i];
    }

    it += this->wi * this->ht + this->bi;

    double zit = it;
    it = Sigmoid(zit);

    // Candidate Cell

    double candidate = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        candidate += this->uc[i] * input[i];
    }

    candidate += this->wc * this->ht + this->bc;

    double zcand = candidate;
    candidate = Tanh(zcand);

    // Cell State

    double ct_1 = this->ct;
    this->ct = ft * ct_1 + it * candidate;

    // Output Gate

    double ot = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        ot += this->weights[i] * input[i];
    }

    ot += this->wo * this->ht + this->bias;

    double zot = ot;
    ot = Sigmoid(zot);

    // Hidden State

    double ht = ot * this->actFunc.Activate(ct);

    // Atualizando histórico

    this->historic.output = ht;
    this->historic.input = input;

    this->historic.steps["ft"] = ft;
    this->historic.steps["it"] = it;
    this->historic.steps["ot"] = ot;

    this->historic.steps["zft"] = zft;
    this->historic.steps["zit"] = zit;
    this->historic.steps["zot"] = zot;
    this->historic.steps["zgt"] = zcand;

    this->historic.steps["ct"] = ct;
    this->historic.steps["ct_1"] = ct_1;

    this->historic.steps["ht_1"] = this->ht;

    this->historic.steps["gt"] = candidate;

    // Atualizando último estado oculto
    this->ht = ht;

    return ht;
}

std::vector<double> LSTMNeuron::Learn(double learningRate, double gradient) {
    // Capturando todos os dados necessários para o backpropagation

    double ft = this->historic.steps["ft"];
    double it = this->historic.steps["it"];
    double ot = this->historic.steps["ot"];

    double zft = this->historic.steps["zft"];
    double zit = this->historic.steps["zit"];
    double zot = this->historic.steps["zot"];
    
    double ct = this->ct;
    double ct_1 = this->historic.steps["ct_1"];

    double ht_1 = this->historic.steps["ht_1"];

    double gt = this->historic.steps["gt"];
    double zgt = this->historic.steps["zgt"];

    std::vector<double> input = this->historic.input;

    // Calculando todos os gradientes parciais

    double outputGr = gradient * this->actFunc.Activate(ct);

    double cellStateGr = gradient * ot * this->actFunc.Derivative(ct);
    double candCellStateGr = cellStateGr * it;

    double inputGr = cellStateGr * gt;
    double forgetGr = cellStateGr * ct_1;

    double d_o = gradient * this->actFunc.Activate(ct) * SigmoidDx(zot);
    double d_f = cellStateGr * ct_1 * SigmoidDx(zft);
    double d_i = cellStateGr * gt * SigmoidDx(zit);
    double d_g = cellStateGr * it * TanhDx(zgt);

    std::vector<double> delta = { d_f, d_i, d_g, d_o };
    
    //

    // Output gate update

    for (int i = 0; i < this->weightsNum; ++i) {
        this->weights[i] -= learningRate * outputGr * SigmoidDx(ot) * input[i];
    }

    this->wo -= learningRate * outputGr * SigmoidDx(ot) * ht_1;
    this->bias -= learningRate * outputGr * SigmoidDx(ot);

    // Input gate update

    for (int i = 0; i < this->weightsNum; ++i) {
        this->ui[i] -= learningRate * inputGr * SigmoidDx(it) * input[i];
    }

    this->wi -= learningRate * inputGr * SigmoidDx(it) * ht_1;
    this->bi -= learningRate * inputGr * SigmoidDx(it);

    // Forget gate update

    for (int i = 0; i < this->weightsNum; ++i) {
        this->uf[i] -= learningRate * forgetGr * SigmoidDx(ft) * input[i];
    }

    this->wf -= learningRate * forgetGr * SigmoidDx(ft) * ht_1;
    this->bf -= learningRate * forgetGr * SigmoidDx(ft);

    // Candidate Cell

    for (int i = 0; i < this->weightsNum; ++i) {
        this->uc[i] -= learningRate * candCellStateGr * TanhDx(gt) * input[i];
    }

    this->wc -= learningRate * candCellStateGr * TanhDx(gt) * ht_1;
    this->bc -= learningRate * candCellStateGr * TanhDx(gt);

    //  

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
