// Definição genérica de Rede Neural

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>

#include <filesystem>
#include <fstream>
#include <cstdlib>

#include <nlohmann/json.hpp>

#include <neural_network/Core/Neuron/BasicNeuron.hpp>
#include <neural_network/Core/Neuron/LSTMNeuron.hpp>

using Json = nlohmann::json;

// Buffer para leitura e escrita de dados (pesos e bias) da rede neural

constexpr std::size_t NEURON_BUFFER_DOUBLES_COUNT = 4096; // Quantidade máxima de doubles guardados por linha
constexpr std::size_t RW_STREAM_NEURON_BUFFER_SIZE = NEURON_BUFFER_DOUBLES_COUNT * sizeof(double); // Memória necessária para guardar a quantia de doubles

// Buffer de leitura e escrita de dados da rede

class StreamNeuronBuffer {
    private:
        char* buffer = nullptr;

    public:
        StreamNeuronBuffer(void) {
            this->buffer = new char[RW_STREAM_NEURON_BUFFER_SIZE];
        }
        
        ~StreamNeuronBuffer() {
            delete[] this->buffer;
        }

        inline char* Get(void) noexcept {
            return this->buffer;
        }

        inline void FlushBuffer(void) noexcept {
            if (!this->buffer) {
                return;
            }

            std::memset(this->buffer, 0, RW_STREAM_NEURON_BUFFER_SIZE);
        }
};

static StreamNeuronBuffer streamNeuronBuffer;

//

std::vector<double> NeuralNetwork::ForwardPass(const std::vector<double>& inputs) {
    std::vector<double> vec, aux;

    // Inputs

    int neurons = this->metadata.architecture[0].neurons;

    vec.resize(neurons);
    vec = { 0 };

    int safeInpSize = (neurons < inputs.size())? neurons : inputs.size();

    for (int i = 0; i < safeInpSize; ++i) {
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

void NeuralNetwork::BackPropagation(const std::vector<double>& expected, const std::string& trainFuncID, double learningRate) {
    std::vector<double> deltas, aux;

    // Calculando para a camada de saída

    int lastLayerID = this->layers.size() - 1;

    NeuralNetworkLayer& layer = this->layers[lastLayerID];
    int layerSize = layer.size();

    deltas.resize(layerSize);
        
    // Percorrendo neurônios na camada de saída

    const TrainingErrorFunctionDx& trainFunc = TRAINING_ERROR_FUNCTION_DX_LIST[trainFuncID];

    for (int j = 0; j < layerSize; ++j) {
        std::shared_ptr<INeuron> neuron = layer[j];

        // Calculando gradiente de erro
        double gradient = trainFunc(neuron->a, expected[j]);

        // Atualizando pesos e bias por neurônio na camada atual
        deltas[j] = neuron->Learn(learningRate, gradient);
    }

    // Calculando para as demais camadas

    for (int i = lastLayerID - 1; i >= 0; --i) {
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
            aux[j] = neuron->Learn(learningRate, gradient);
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
    
    info.open(info_file);
    param.open(param_file, std::ios::binary);

    // Erro ao abrir/criar os arquivos

    if (!(info.is_open() && param.is_open())) {
        return false;
    }

    // Salvando informações da rede neural

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

    for (const NeuralNetworkLayer& layer : neuralNetwork->layers) {
        for (const std::shared_ptr<INeuron>& neuron : layer) {
            const INeuronBuffer buffer = neuron->Serialize();
            
            // Estipulando o tamanho seguro para evitar overflow
            std::size_t buffer_size = (buffer.size > RW_STREAM_NEURON_BUFFER_SIZE)? RW_STREAM_NEURON_BUFFER_SIZE : buffer.size;

            // Copia o buffer do neurônio para um buffer fixo de caracteres

            streamNeuronBuffer.FlushBuffer();

            std::memcpy(
                streamNeuronBuffer.Get(),
                buffer.bytes, buffer_size
            );

            // Copia o buffer de caracteres no arquivo. Todo neurônio ocupa a mesma quantidade de caracteres por linha.

            param.write(streamNeuronBuffer.Get(), RW_STREAM_NEURON_BUFFER_SIZE);

            delete[] buffer.bytes;
        }
    }

    //

    param.close();

    //

    return true;
}

bool NeuralNetwork::SaveNeuralNetwork(const std::shared_ptr<NeuralNetwork>& neuralNetwork) {
    return NeuralNetwork::SaveNeuralNetwork(neuralNetwork, neuralNetwork->metadata.name);
}

std::shared_ptr<NeuralNetwork> NeuralNetwork::LoadNeuralNetwork(const std::string& name) {
    std::shared_ptr<NeuralNetwork> neuralNetwork = CreateNeuralNetwork();

    std::string param_file = std::string(NEURAL_NETWORKS_DIR) + name + "/" + "param.dat";
    std::string info_file = std::string(NEURAL_NETWORKS_DIR) + name + "/" + "info.json";

    // Verificando se os arquivos existem

    if (!(std::filesystem::exists(info_file) && std::filesystem::exists(param_file))) {
        return nullptr;
    }

    std::ifstream info, param;

    info.open(info_file);
    param.open(param_file, std::ios::binary);

    // Verificando se o arquivo foi aberto com sucesso

    if (!(info.is_open() && param.is_open())) {
        return nullptr;
    }

    // Salvando informações da rede neural

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

    // Construindo as camadas de neurônios na memória

    int amountOfInputs = 1;

    for (const NeuralNetworkArchitectureData& data : metadata.architecture) {
        const std::string actFuncID = data.activationFunction;
        const int amountOfNeurons = data.neurons; 
        const std::string neuronType = data.type;

        if (neuronType == "basic") {
            NeuralNetwork::CreateNeuronLayer<BasicNeuron>(
                neuralNetwork, actFuncID,
                amountOfNeurons, amountOfInputs
            );
        } else if (neuronType == "lstm") {
            NeuralNetwork::CreateNeuronLayer<LSTMNeuron>(
                neuralNetwork, actFuncID,
                amountOfNeurons, amountOfInputs
            );
        }

        amountOfInputs = amountOfNeurons;
    }

    // Deserializando os pesos para cada camada

    for (NeuralNetworkLayer& layer : neuralNetwork->layers) {
        for (const std::shared_ptr<INeuron>& neuron : layer) {
            // Validando se há linhas disponíveis para criação da rede. Caso não tenha, retorna a rede neural construída

            streamNeuronBuffer.FlushBuffer();

            if (!param.read(streamNeuronBuffer.Get(), RW_STREAM_NEURON_BUFFER_SIZE)) {
                param.close();
                return neuralNetwork;
            }

            const double* buffer = reinterpret_cast<const double*>(streamNeuronBuffer.Get());
            neuron->Deserialize(buffer); // o neurônio deserializa os dados para seus pesos e biases
        }
    }

    //

    param.close();

    //

    return neuralNetwork;
}
