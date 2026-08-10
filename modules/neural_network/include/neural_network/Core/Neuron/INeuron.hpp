// Definição da interface de um neurônio na rede

#pragma once

#include <memory>
#include <type_traits>
#include <functional>
#include <cstddef>
#include <vector>
#include <string>
#include <map>

#include <neural_network/Core/NeuralNetwork/definitions.hpp>

// Definições

constexpr double STANDARD_MAX_ABSOLUTE_WEIGHT = 0.1f; // Weight padrão na inicialização
constexpr double STANDAND_MAX_ABSOLUTE_BIAS = 0.1f;  // Bias padrão na inicialização

// Funções de ativação e derivadas do neurônio

struct INeuronActivationFunctions {
    INeuronActivationFunction Activate;
    INeuronActivationFunctionDx Derivative;
};

// Estrutura de bytes que representa os pesos e biases do neurônio
struct INeuronBuffer {
    const double* bytes;
    std::size_t size;
};

//

class INeuron {
    public:
        // Quantidade de pesos
        int weightsNum;

        // Pesos da rede
        std::vector<double> weights;

        // Viés
        double bias;

        // Dados de execução
        std::vector<double> x; // Última entrada
        double z, a; // Última saída linear / última saída ativada

        // Função de ativação do neurônio e sua derivada
        INeuronActivationFunctions actFunc;

        explicit INeuron(int weightsNum);
        INeuron(void) = default;

        ~INeuron() = default;

        virtual double Load(const std::vector<double>& input) = 0;
        virtual double Learn(double learningRate, double gradient) = 0;

        //
        
        virtual const std::vector<double> Weights(int ref) const = 0; // Retorna todos os pesos que estão associados a determinada ordem de entrada (neurônio anterior)
        
        // Funções auxiliares

        virtual const INeuronBuffer Serialize(void) const noexcept = 0; // Gera um buffer que representa todos os pesos e biases do neurônio
        virtual void Deserialize(const double* buffer) noexcept = 0; // Carrega o buffer de pesos e biases do neurônio
};

template <class NeuronType>
inline std::shared_ptr<INeuron> CreateNeuron(int weights) {
    static_assert(std::is_base_of_v<INeuron, NeuronType>, "NeuronType must be derived from INeuron.");

    std::shared_ptr<NeuronType> neuron = std::make_shared<NeuronType>(weights);
    return std::dynamic_pointer_cast<INeuron>(neuron);
}
