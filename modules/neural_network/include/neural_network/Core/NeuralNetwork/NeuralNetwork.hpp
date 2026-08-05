// Definição genérica de Rede Neural

#pragma once

#include <memory>
#include <vector>
#include <string>

#include "neural_network/Core/NeuralNetwork/INeuron.hpp"
#include "neural_network/Core/NeuralNetwork/ErrorFunction.hpp"

//

// Taxa de aprendizado na rede nos treinamentos
constexpr double LEARNING_RATE = 0.1f;

// Definição de camada

using NeuralNetworkLayer = std::vector<std::shared_ptr<INeuron>>;

// Definição dos metadados da rede

// Estrutura de metadados
struct NeuralNetworkArchitectureData {
    int neurons;
    std::string type;
    std::string activationFunction;
};

// Container de metadados da rede
using NeuralNetworkArchitecture = std::vector<NeuralNetworkArchitectureData>;

// Estrutura de metadados da rede neural
struct NeuralNetworkMetadata {
    std::string name;
    std::string createdAt;

    // Capacidade nominal de acerto da rede
    float fitness;

    // Pares de moedas já treinados
    std::vector<std::string> pairsTrained;

    // Arquitetura da rede
    NeuralNetworkArchitecture architecture;
};

// Histórico de execução da rede

struct NeuralNetworkData {
    struct NeuronData {
        std::vector<double> x; // valores dos últimos inputs nesse neurônio
        double z, a; // valor antes da ativação, valor depois da ativação, produzidos por um neurônio específico
    };

    // Valores armazenados por camada

    std::vector<
        std::vector<NeuronData>
    > layers;
};

// Definição de uma rede neural geral

class NeuralNetwork {
    public:
        // Metadados
        NeuralNetworkMetadata metadata;

        // Camadas da rede
        std::vector<NeuralNetworkLayer> layers;

        // Valores gerados pela rede na última execução, por camada e neurônio
        NeuralNetworkData data;

        NeuralNetwork(void) = default;
        ~NeuralNetwork() = default;

        // Executa a rede e gera o resultado
        std::vector<double> ForwardPass(const std::vector<double>& inputs);

        // Realiza o treinamento da rede
        void BackPropagation(const std::vector<double>& expected, const ErrorFunction& errorFunction);
};
