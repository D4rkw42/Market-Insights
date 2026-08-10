// Definição genérica de Rede Neural

#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <functional>
#include <type_traits>

#include <neural_network/Core/NeuralNetwork/definitions.hpp>
#include <neural_network/Core/Neuron/INeuron.hpp>

//

// Diretório das redes neurais
constexpr const char* NEURAL_NETWORKS_DIR = "data/neural_networks/";

// Definição de camada

using NeuralNetworkLayer = std::vector<std::shared_ptr<INeuron>>;

// Taxa de aprendizado padrão da rede
constexpr double DEFAULT_LEARNING_RATE = 0.01;

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
        void BackPropagation(const std::vector<double>& expected, const std::string& trainFuncID, double learningRate = DEFAULT_LEARNING_RATE);

        // Utility

        // Salva uma rede neural na memória

        static bool SaveNeuralNetwork(const std::shared_ptr<NeuralNetwork>& neuralNetwork, const std::string& name);
        static bool SaveNeuralNetwork(const std::shared_ptr<NeuralNetwork>& neuralNetwork);

        // Carrega uma rede neural. Retorna nullptr caso não exista
        static std::shared_ptr<NeuralNetwork> LoadNeuralNetwork(const std::string& name);

    private:
        // Adiciona umna camada de neurônios na rede
        template <class INeuronType>
        static void CreateNeuronLayer(const std::shared_ptr<NeuralNetwork>& neuralNetwork, const std::string& actFuncID, int amountOfNeurons, int amountOfInputs) {
            NeuralNetworkLayer layer;

            for (int i = 0; i < amountOfNeurons; ++i) {
                std::shared_ptr<INeuron> neuron = CreateNeuron<INeuronType>(amountOfInputs);

                neuron->actFunc.Activate = ACTIVATION_FUNCTION_LIST[actFuncID];
                neuron->actFunc.Derivative = ACTIVATION_FUNCTION_DX_LIST[actFuncID];

                layer.push_back(neuron);
            }

            neuralNetwork->layers.push_back(layer);
        }
};

inline std::shared_ptr<NeuralNetwork> CreateNeuralNetwork(void) {
    return std::make_shared<NeuralNetwork>();
}
