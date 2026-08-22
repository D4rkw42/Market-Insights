// Define um neurônio comum na rede neural

#include <neural_network/Core/Neuron/BasicNeuron.hpp>

BasicNeuron::BasicNeuron(int weightNum) : INeuron(weightNum) {}

//

double BasicNeuron::Load(const std::vector<double>& input) {
    double z = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        z += this->weights[i] * input[i];
    }

    z += this->bias;

    double output = this->actFunc.Activate(z);

    // Atualizando histórico

    this->historic.input = input;
    this->historic.output = output;

    return output;
}

INeuronGradientsAndDeltas BasicNeuron::MakeGradientsAndDeltas(double gradient) {
    INeuronGradientsAndDeltas obj;

    double g = gradient * this->actFunc.Derivative(this->historic.output);

    obj.gradients = std::vector<double> { g };
    obj.deltas = obj.gradients;

    return obj;
}

void BasicNeuron::Learn(const std::vector<double>& gradients, double learningRate) {
    double delta = gradients[0];

    for (int i = 0; i < this->weightsNum; ++i) {
        this->weights[i] -= learningRate * delta * this->historic.input[i];
    }

    this->bias -= learningRate * delta;
}

const std::vector<double> BasicNeuron::Weights(int ref) const {
    return std::vector<double> { this->weights[ref] };
}

//

const INeuronBuffer BasicNeuron::Serialize(void) const noexcept {
    std::size_t elements = this->weightsNum + 1;

    double* buffer = new double[elements];
    std::size_t size = elements * sizeof(double);
    
    int i;

    for (i = 0; i < this->weightsNum; ++i) {
        buffer[i] = this->weights[i];
    }

    buffer[i] = this->bias;

    return INeuronBuffer { buffer, size };
}

void BasicNeuron::Deserialize(const double* buffer) noexcept {
    int i;

    for (i = 0; i < this->weightsNum; ++i) {
        this->weights[i] = buffer[i];
    }

    this->bias = buffer[i];
}
