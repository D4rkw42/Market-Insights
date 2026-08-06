// Define um neurônio LSTM na rede neural

#pragma once

#include "neural_network/Core/NeuralNetwork/INeuron.hpp"

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

        // Cell States
        double
            ct = 0,
            ct_1 = 0;

        // Hidden States
        double 
            ht = 0,
            ht_1 = 0;

    public:
        LSTMNeuron(int weightsNum);
        LSTMNeuron(void) = default;

        virtual ~LSTMNeuron() = default;

        double Load(const std::vector<double>& input) override;
        void UpdateWeightsAndBias(const std::vector<double>& lastInput, const double delta, const double learningRate) override;
};
