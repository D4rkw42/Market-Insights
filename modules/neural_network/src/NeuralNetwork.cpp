// Definição genérica de Rede Neural

#include "neural_network/global.hpp"

#include "neural_network/Core/NeuralNetwork/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(void) {}

//

std::vector<double> NeuralNetwork::ForwardPass(const std::vector<double>& inputs) {
    std::vector<double> vec, aux;

    // Inputs

    vec.resize(inputs.size());

    for (int i = 0; i < inputs.size(); ++i) {
        std::shared_ptr<INeuron> neuron = this->layers[0][i];
        vec[i] = neuron->Load(std::vector<double> { inputs[i] });
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

void NeuralNetwork::BackPropagation(double loss, double learningRate) {

}
