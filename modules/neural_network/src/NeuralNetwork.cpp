// Definição genérica de Rede Neural

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include <neural_network/definitions.hpp>

#include <neural_network/Core/Neuron/BasicNeuron.hpp>
#include <neural_network/Core/Neuron/LSTMNeuron.hpp>

using Json = nlohmann::json;

//

std::vector<double> NeuralNetwork::ForwardPass(const std::vector<double>& inputs) {
    std::vector<double> vec, aux;

    // Inputs
    
    int neurons = this->metadata.architecture[0].neurons;

    vec.resize(neurons);

    for (int i = 0; i < neurons; ++i) {
        std::shared_ptr<INeuron> neuron = this->layers[0][i];
        vec[i] = neuron->Load(std::vector<double> { inputs[i] });
    }

    // Hidden & Outputs

    for (int i = 1; i < this->layers.size(); ++i) {
        neurons = this->metadata.architecture[i].neurons;
        aux.resize(neurons);

        // Obtendo resultado de cada operação por camada

        for (int j = 0; j < neurons; ++j) {
            std::shared_ptr<INeuron> neuron = this->layers[i][j];
            aux[j] = neuron->Load(vec);
        }

        vec = std::move(aux);
    }

    return vec;
}

void NeuralNetwork::BackPropagation(const std::vector<double>& expected, const TrainingErrorFunctionDx& trainingErrorFunction) {
    std::vector<double> deltas, aux;

    // Calculando para a camada de saída

    int lastLayerID = this->layers.size() - 1;

    NeuralNetworkLayer& layer = this->layers[lastLayerID];
    int layerSize = layer.size();

    deltas.resize(layerSize);
        
    // Percorrendo neurônios na camada de saída

    for (int j = 0; j < layerSize; ++j) {
        std::shared_ptr<INeuron> neuron = layer[j];

        // Calculando gradiente de erro
        double gradient = trainingErrorFunction(neuron->a, expected[j]);

        // Atualizando pesos e bias por neurônio na camada atual
        deltas[j] = neuron->Learn(LEARNING_RATE, gradient);
    }

    // Calculando para as demais camadas

    for (int i = this->layers.size() - 2; i > 0; --i) {
        NeuralNetworkLayer& posLayer = this->layers[i + 1];
        NeuralNetworkLayer& layer = this->layers[i];

        int layerSize = layer.size();

        aux.resize(layerSize);

        //

        for (int j = 0; j < layerSize; ++j) {        
            std::shared_ptr<INeuron> neuron = layer[j];

            double gradient = 0;

            // Calculando gradiente com base na camada posterior e deltas anteriores

            for (int k = 0; k < posLayer.size(); ++k) {
                // Um neurônio pode ter mais de um peso que se relaciona com a entrada
                const std::vector<double> weights = posLayer[k]->Weights(j);
                double partialGradient = 0;

                for (int l = 0; l < weights.size(); ++l) {
                    partialGradient += weights[l];
                }

                gradient += partialGradient * deltas[k];
            }

            // Atualizando pesos e bias para o neurônio atual e obtendo delta
            aux[j] = neuron->Learn(LEARNING_RATE, gradient);
        }

        deltas = aux;
    }
}

// Utility

bool NeuralNetwork::SaveNeuralNetwork(const std::shared_ptr<NeuralNetwork>& neuralNetwork, const std::string& name) {
    std::string param_path = std::string(NEURAL_NETWORKS_DIR) + name + "/";
    std::string info_path = std::string(NEURAL_NETWORKS_DIR) + name + "/";

    std::string param_file = param_path + "param.dat";
    std::string info_file = info_path + "info.json";

    std::ofstream info, param;

    // Criando diretórios caso não existam

    if (!std::filesystem::exists(param_path)) {
        std::filesystem::create_directories(param_path);
    }

    if (!std::filesystem::exists(info_path)) {
        std::filesystem::create_directories(info_path);
    }

    // Salvando informações da rede neural

    info.open(info_file);

    //

    const NeuralNetworkMetadata& metadata = neuralNetwork->metadata;

    Json jObject;

    jObject["name"] = metadata.name;
    jObject["created_at"] = metadata.createdAt;

    jObject["fitness"] = static_cast<int>(metadata.fitness * 100) * 0.01;

    Json architecture {};

    for (int i = 0; i < metadata.architecture.size(); ++i) {
        const NeuralNetworkArchitectureData& d = metadata.architecture[i];

        Json data;

        data["neurons"] = d.neurons;
        data["type"] = d.type;
        data["activation_function"] = d.activationFunction;

        architecture[i] = data;
    }

    jObject["architecture"] = architecture;
    jObject["pairs_trained"] = metadata.pairsTrained;

    // Salvando dados do JSON no arquivo
    std::string data = jObject.dump(4);
    info << data;

    //

    info.close();

    // Salvando parâmetros da rede neural

    param.open(param_file);

    //

    for (const NeuralNetworkLayer& layer : neuralNetwork->layers) {
        for (const std::shared_ptr<INeuron>& neuron : layer) {
            const INeuronBuffer buffer = neuron->Serialize();

            const char* buffer_char = reinterpret_cast<const char*>(buffer.bytes);
            param.write(buffer_char, buffer.size);

            delete[] buffer.bytes;

            param << '\n';
        }
    }

    //

    param.close();

    //

    return true;
}

std::shared_ptr<NeuralNetwork> NeuralNetwork::LoadNeuralNetwork(const std::string& name) {
    std::shared_ptr<NeuralNetwork> neuralNetwork = CreateNeuralNetwork();

    std::string param_file = std::string(NEURAL_NETWORKS_DIR) + name + "/" + "param.dat";
    std::string info_file = std::string(NEURAL_NETWORKS_DIR) + name + "/" + "info.json";

    std::ifstream info, param;

    // Salvando informações da rede neural

    info.open(info_file);

    //

    Json jObject;
    info >> jObject;

    NeuralNetworkMetadata& metadata = neuralNetwork->metadata;

    metadata.name = jObject["name"];
    metadata.createdAt = jObject["created_at"];

    metadata.fitness = jObject["fitness"];

    for (const Json& object : jObject["architecture"]) {
        NeuralNetworkArchitectureData data;

        data.neurons = object["neurons"];
        data.type = object["type"];
        data.activationFunction = object["activation_function"];

        metadata.architecture.push_back(data);
    }

    metadata.pairsTrained = jObject["pairs_trained"];

    //

    info.close();

    // Salvando parâmetros da rede neural

    param.open(param_file);

    //

    // Construindo as camadas de neurônios na memória

    int amountOfInputs = 1;

    for (const NeuralNetworkArchitectureData& data : metadata.architecture) {
        const char* activationFunctionName = data.activationFunction.c_str();
        const int amountOfNeurons = data.neurons; 
        const std::string neuronType = data.type;

        INeuronActivationFunctions functions;

        functions.Activate = ACTIVATION_FUNCTION_LIST[activationFunctionName];
        functions.Derivative = ACTIVATION_FUNCTION_DX_LIST[activationFunctionName];

        if (neuronType == "basic") {
            NeuralNetwork::CreateNeuronLayer<BasicNeuron>(neuralNetwork, functions, amountOfNeurons, amountOfInputs);
        } else if (neuronType == "lstm") {
            NeuralNetwork::CreateNeuronLayer<LSTMNeuron>(neuralNetwork, functions, amountOfNeurons, amountOfInputs);
        }

        amountOfInputs = amountOfNeurons;
    }

    // Deserializando os pesos para cada camada

    char streamLine[2048];

    for (NeuralNetworkLayer& layer : neuralNetwork->layers) {
        for (const std::shared_ptr<INeuron>& neuron : layer) {
            // Validando se há linhas disponíveis para criação da rede. Caso não tenha, retorna a rede neural construída
            if (!param.getline(streamLine, 2048, '\n')) {
                param.close();
                return neuralNetwork;
            }

            const double* buffer = reinterpret_cast<const double*>(&streamLine[0]);
            neuron->Deserialize(buffer); // o neurônio deserializa os dados para seus pesos e biases
        }
    }

    //

    param.close();

    //

    return neuralNetwork;
}
