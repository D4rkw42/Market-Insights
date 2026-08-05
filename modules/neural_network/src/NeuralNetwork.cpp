// Definição genérica de Rede Neural

#include "neural_network/Core/NeuralNetwork/NeuralNetwork.hpp"

#include "neural_network/global.hpp"

//

std::vector<double> NeuralNetwork::ForwardPass(const std::vector<double>& inputs) {
    std::vector<double> vec, aux;

    // Inputs

    vec.resize(inputs.size());

    for (int i = 0; i < inputs.size(); ++i) {
        std::shared_ptr<INeuron> neuron = this->layers[0][i];
        vec[i] = neuron->Load(std::vector<double> { inputs[i] });

        // Copiando últimos valores da entrada
        this->data.layers[0][i].x = inputs;
    }

    // Hidden & Outputs

    for (int i = 1; i < this->layers.size(); ++i) {
        int neurons = this->metadata.architecture[i].neurons;
        aux.resize(neurons);

        // Obtendo resultado de cada operação por camada

        for (int j = 0; j < neurons; ++j) {
            std::shared_ptr<INeuron> neuron = this->layers[i][j];
            aux[j] = neuron->Load(vec);

            // Atualizando registro
            this->data.layers[i][j].z = aux[j];

            // Atualizando registro da entrada
            this->data.layers[i][j].x = vec;
        }

        // Aplicando a função de ativação da camada

        const std::string actFuncName = this->metadata.architecture[i].activationFunction;
        ActivationFunction actFunc = neuralNetworkActivationFunctions[actFuncName.c_str()];

        for (int j = 0; j < neurons; ++j) {
            aux[j] = actFunc.CalculateFromLaw(aux[j]);

            // Atualizando registro
            this->data.layers[i][j].a = aux[j];
        }

        vec = std::move(aux);
    }

    return vec;
}

void NeuralNetwork::BackPropagation(const std::vector<double>& output, const std::vector<double>& expected, ErrorFunction& errorFunction) {
    std::vector<double> deltas, aux;
    std::vector<double> lastOutput = output;

    // Calculando para a camada de saída

    int lastLayerID = this->layers.size() - 1;

    NeuralNetworkLayer& layer = this->layers[lastLayerID];

    const std::string actFuncName = this->metadata.architecture[lastLayerID].activationFunction;
    ActivationFunction actFunc = neuralNetworkActivationFunctions[actFuncName.c_str()];

    deltas.resize(layer.size());
        
    // Percorrendo neurônios na camada de saída

    for (int j = 0; j < layer.size(); ++j) {
        std::vector<double> x = this->data.layers[lastLayerID][j].x;

        double z = this->data.layers[lastLayerID][j].z;
        double a = this->data.layers[lastLayerID][j].a;

        // Delta para cada neurônio na camada atual
        double gradient = errorFunction.CalculateFromDerivative(lastOutput[j], expected[j]);
        deltas[j] = gradient * actFunc.CalculateFromDerivative(a);

        // Atualizando pesos e bias por neurônio na camada atual

        std::shared_ptr<INeuron> neuron = layer[j];
        neuron->UpdateWeightsAndBias(x, deltas[j], LEARNING_RATE);
    }

    // Calculando para as demais camadas

    for (int i = this->layers.size() - 2; i > 0; --i) {
        NeuralNetworkLayer& layer = this->layers[i];

        const std::string actFuncName = this->metadata.architecture[i].activationFunction;
        ActivationFunction actFunc = neuralNetworkActivationFunctions[actFuncName.c_str()];

        NeuralNetworkLayer& posLayer = this->layers[i + 1];

        aux.resize(layer.size());

        //

        for (int j = 0; j < layer.size(); ++j) {
            std::vector<double> x = this->data.layers[i][j].x;
            
            double z = this->data.layers[i][j].z;
            double a = this->data.layers[i][j].a;
            
            double gradient = 0;

            // Calculando gradiente no neurônio com base nos deltas da camada posterior

            for (int k = 0; k < posLayer.size(); ++k) {
                gradient += deltas[k] * posLayer[k]->weights[j];
            }

            // Calculando novos deltas

            aux[j] = gradient * actFunc.CalculateFromDerivative(a);


            // Atualizando pesos e bias para o neurônio atual

            std::shared_ptr<INeuron> neuron = layer[j];
            neuron->UpdateWeightsAndBias(x, aux[j], LEARNING_RATE);
        }

        //

        deltas = aux;
    }
}
