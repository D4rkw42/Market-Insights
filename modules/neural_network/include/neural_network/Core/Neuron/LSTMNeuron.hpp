// Define um neurônio LSTM na rede neural

#pragma once

#include <neural_network/Core/Neuron/INeuron.hpp>

class LSTMNeuron : public INeuron {
    private:
        // Pesos especiais para a célula LSTM

        // Forget Gate
        std::vector<double> uf;
        double wf, bf;

        // Input Gate
        std::vector<double> ui;
        double wi, bi;

        // Output Gate // Demais pesos atribuídos à INeuron
        double wo;

        // Candidate Cell
        std::vector<double> uc;
        double wc, bc;

        // Cell State (previous)
        double ct_1 = 0;

        // Hidden State (previous)
        double ht_1 = 0;

        // Estados da célula para treinamento
        double ft, it, ot, candidate;

    public:
        LSTMNeuron(int weightsNum);
        LSTMNeuron(void) = default;

        virtual ~LSTMNeuron() = default;

        double Load(const std::vector<double>& input) override;
        double Learn(double learningRate, double gradient) override;

        const std::vector<double> Weights(int ref) const override;
};
