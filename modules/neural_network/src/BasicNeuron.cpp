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
    double delta = this->actFunc.Derivative(this->z) * gradient;

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
