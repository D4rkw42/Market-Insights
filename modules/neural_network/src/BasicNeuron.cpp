// Define um neurônio comum na rede neural

#include <neural_network/Core/Neuron/BasicNeuron.hpp>

BasicNeuron::BasicNeuron(int weightNum) : INeuron(weightNum) {}

//

double BasicNeuron::Load(const std::vector<double>& input) {
    this->x = input;
    this->z = 0;

    for (int i = 0; i < this->weightsNum; ++i) {
        this->z += this->weights[i] * input[i];
    }

    this->z += this->bias;
    this->a = this->actFunc.Activate(this->z);

    return this->a;
}

double BasicNeuron::Learn(double learningRate, double gradient) {
    double delta = this->actFunc.Derivative(this->z);

    // Atualizando pesos

    for (int i = 0; i < this->weightsNum; ++i) {
        this->weights[i] -= learningRate * delta * this->x[i];
    }

    // Atualizando bias
    this->bias -= learningRate * delta;

    return delta;
}

const std::vector<double> BasicNeuron::Weights(int ref) const {
    return std::vector<double> { this->weights[ref] };
}

//

const INeuronBuffer BasicNeuron::Serialize(void) const noexcept {
    double* mem = new double[this->weightsNum + 1];
    std::vector<double> buffer;
    
    buffer.reserve(this->weightsNum + 1);
    buffer.insert(buffer.end(), this->weights.begin(), this->weights.end());
    buffer.push_back(this->bias);

    for (int i = 0; i < buffer.size(); ++i) {
        mem[i] = buffer[i];
    }

    return INeuronBuffer { mem, buffer.size() };
}

void BasicNeuron::Deserialize(const double* buffer) noexcept {
    int i;

    for (i = 0; i < this->weightsNum; ++i) {
        this->weights[i] = buffer[i];
    }

    this->bias = buffer[i];
}
