// Representação de uma rede neural para criação de redes novas

#pragma once

#include <memory>
#include <vector>
#include <tuple>
#include <string>

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>

// Definição de um esquema de camada

using NeuralNetworkLayerDescriptor = std::tuple<int, std::string, std::string>; // Número de neurônios, tipo de neurônio, função de ativação
using NeuralNetworkLayerSchemas = std::vector<NeuralNetworkLayerDescriptor>;

//

class NeuralNetworkSchema {
    private:
        NeuralNetworkLayerSchemas layerSchemas;

    public:
        NeuralNetworkSchema(void) = default;
        ~NeuralNetworkSchema() = default;

        // Adição e remoção de esquemas de camada

        void CreateLayerSchemaAt(int neurons, const std::string& type, const std::string& activationFunction, int id); // Adiciona um esquema na posição especificada ou no final pos ultrapasse o tamanho do vetor de esquemas
        void CreateLayerSchema(int neurons, const std::string& type, const std::string& activationFunction); // Adiciona um esquema na última posição

        void RemoveLayerSchemaAt(int id); // Remove o esquema da posição especificada
        void RemoveLayerSchema(void); // Remove o esquema da última posição

        // Gera uma nova rede neural com base nas informações especificadas. Retorna nullptr caso não seja possível sua criação.
        std::shared_ptr<NeuralNetwork> GenerateNeuralNetwork(const std::string& name) const;

        // Retorna a quantidade de schemas criados  

        inline int GetAmountOfSchemas(void) const noexcept {
            return this->layerSchemas.size();
        }
};
