// Definição da interface de um neurônio na rede

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

class INeuron {
    public:
        // Quantidade de pesos
        int weightsNum;

        // Pesos da rede
        std::vector<double> weights;

        // Viés
        double bias = 0.0f;

        INeuron(int weightsNum);
        ~INeuron() = default;

        virtual double Load(const std::vector<double>& input) const = 0;
};

template <class NeuronType>
inline std::shared_ptr<INeuron> CreateNeuron(int weights) {
    static_assert(std::is_base_of_v(INeuron, NeuronType), "NeuronType must be derived from INeuron.");

    std::shared_ptr<NeuronType> neuron = std::make_shared<NeuronType>(weights);
    return std::dynamic_pointer_cast<INeuron>(neuron);
}
