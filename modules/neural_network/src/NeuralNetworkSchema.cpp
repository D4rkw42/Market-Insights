// Representação de uma rede neural para criação de redes novas

#include <neural_network/Core/NeuralNetwork/NeuralNetworkSchema.hpp>

#include <ctime>
#include <iomanip>
#include <sstream>

#include <neural_network/Core/Neuron/BasicNeuron.hpp>
#include <neural_network/Core/Neuron/LSTMNeuron.hpp>

void NeuralNetworkSchema::CreateLayerSchemaAt(int neurons, const std::string& type, const std::string& activationFunction, int id) noexcept {
    NeuralNetworkLayerDescriptor descriptor = { neurons, type, activationFunction };

    int amountOfSchemas = GetAmountOfSchemas();

    // Proteção para a lista de schemas vazia

    if (amountOfSchemas == 0) {
        return;
    }

    // Proteção contra id inválido

    int safeID = (id >= amountOfSchemas)? amountOfSchemas - 1 : id;
    safeID = (safeID < 0)? 0 : safeID;

    auto start = this->layerSchemas.begin();
    auto index = std::next(start, safeID);

    this->layerSchemas.insert(index, descriptor);
}

void NeuralNetworkSchema::CreateLayerSchema(int neurons, const std::string& type, const std::string& activationFunction) noexcept {
    NeuralNetworkLayerDescriptor descriptor = { neurons, type, activationFunction };
    this->layerSchemas.push_back(descriptor);
}

//

void NeuralNetworkSchema::RemoveLayerSchemaAt(int id) noexcept {
    int amountOfSchemas = GetAmountOfSchemas();

    // Proteção para a lista de schemas vazia

    if (amountOfSchemas == 0) {
        return;
    }

    // Proteção contra id inválido

    int safeID = (id >= amountOfSchemas)? amountOfSchemas - 1 : id;
    safeID = (safeID < 0)? 0 : safeID;

    auto start = this->layerSchemas.begin();
    auto index = std::next(start, safeID);

    this->layerSchemas.erase(index);
}

void NeuralNetworkSchema::RemoveLayerSchema(void) noexcept {
    this->layerSchemas.pop_back();
}

//

std::shared_ptr<NeuralNetwork> NeuralNetworkSchema::GenerateNeuralNetwork(const std::string& name) const noexcept {
    if (GetAmountOfSchemas() < 2) {
        return nullptr;
    }

    std::shared_ptr<NeuralNetwork> neuralNetwork = CreateNeuralNetwork();
    
    // Nome da rede neural

    neuralNetwork->metadata.name = name;

    // Data de criação

    auto time = std::time(0);
    auto localTime = *std::localtime(&time);

    std::ostringstream formatedTime;
    
    formatedTime << std::put_time(&localTime, "%d-%m-%Y %H:%M:%S");

    std::string createdAt = formatedTime.str();

    neuralNetwork->metadata.createdAt = createdAt;

    // Percorre todos os schemas para descrever a rede e criar as camadas de neurônios

    NeuralNetworkMetadata& metadata = neuralNetwork->metadata;

    int amountOfInputs = 1;

    for (const NeuralNetworkLayerDescriptor& descriptor : this->layerSchemas) {
        int neurons = std::get<0>(descriptor);
        std::string type = std::get<1>(descriptor);
        std::string activationFunction = std::get<2>(descriptor);

        // Adiciona as informações de cada camada na estrutura da rede

        NeuralNetworkArchitectureData data = { neurons, type, activationFunction };
        metadata.architecture.push_back(data);

        // Realiza a criação inicial de neurônios

        if (type == "basic") {
            NeuralNetwork::CreateNeuronLayer<BasicNeuron>(neuralNetwork, activationFunction, neurons, amountOfInputs);
        } else if (type == "lstm") {
            NeuralNetwork::CreateNeuronLayer<LSTMNeuron>(neuralNetwork, activationFunction, neurons, amountOfInputs);
        }

        amountOfInputs = neurons;
    }

    return neuralNetwork;
}
