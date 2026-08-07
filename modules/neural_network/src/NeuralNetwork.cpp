// Definição genérica de Rede Neural

#include <neural_network/Core/NeuralNetwork/NeuralNetwork.hpp>

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
