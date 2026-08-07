// Definição genérica de Rede Neural

#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>

#include <neural_network/Core/Neuron/INeuron.hpp>

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

// Definição de uma função de erro
using TrainingErrorFunction = std::function<double(const std::vector<double>&, const std::vector<double>&)>;
using TrainingErrorFunctionDx = std::function<double(double, double)>;

// Lista de funções de erro
using TrainingErrorFunctionList = std::map<const char*, const TrainingErrorFunction>;
using TrainingErrorFunctionDxList = std::map<const char*, const TrainingErrorFunctionDx>;

// Definição de uma rede neural geral

class NeuralNetwork {
    public:
        // Metadados
        NeuralNetworkMetadata metadata;

        // Camadas da rede
        std::vector<NeuralNetworkLayer> layers;

        NeuralNetwork(void) = default;
        ~NeuralNetwork() = default;

        // Executa a rede e gera o resultado
        std::vector<double> ForwardPass(const std::vector<double>& inputs);

        // Realiza o treinamento da rede
        void BackPropagation(const std::vector<double>& expected, const TrainingErrorFunctionDx& trainingErrorFunction);
};
