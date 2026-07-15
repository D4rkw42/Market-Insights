// Funções de ativação

#pragma once

#include <cmath>
#include <vector>

#include "neural_network/Math/math.hpp"

// Sigmoid

inline const double Sigmoid(double x) {
    return 1.0 / (1 + std::exp(-x));
}

inline const double SigmoidDx(double x) {
    double sigX = Sigmoid(x);
    return sigX * (1 - sigX);
}

// Tangente Hiperbólica

inline const double Tanh(double x) {
    return std::tanh(x);
}

inline const double TanhDx(double x) {
    double tanhX = Tanh(x);
    return 1 - tanhX * tanhX;
}

// Função Identidade

constexpr const double Identity(double x) {
    return x;
}

constexpr const double IdentityDx(double x) {
    return 1;
}

// LeakyReLu

inline const double LeakyReLu100(double x) {
    return (x > 0)? x : x / 100;
}

inline const double LeakyReLu100Dx(double x) {
    return (x > 0)? 1 : 0.01f;
}

inline const double LeakyReLu1000(double x) {
    return (x > 0)? x : x / 1000;
}

inline const double LeakyReLu1000Dx(double x) {
    return (x > 0)? 1 : 0.001f;
}

// Gaussian Error Linear Unit

inline const double GELU(double x) {
    static const double a = std::sqrt(2.0 / PI);
    return (x * 0.5f) * (1 + Tanh(a * (x + 0.044715 * std::pow(x, 3))));
}

inline const double GELUDx(double x) {
    static const double a = std::sqrt(2.0 / PI);
    const double u = a * (x + 0.044715 * std::pow(x, 3));

    const double TanhU = Tanh(u);

    return 0.5 * (1 + TanhU) + (x * 0.5f) * (1 - TanhU * TanhU) * a * (1 + 0.134145 * x * x);
}

// Softmax

inline const std::vector<double> SoftMax(const std::vector<double>& logits) {
    std::vector<double> vec = logits;

    double sum = 0;

    for (double l : logits) {
        sum += std::exp(l);
    }

    for (int i = 0; i < vec.size(); ++i) {
        vec[i] = vec[i] / sum;
    }

    return vec;
}

inline const std::vector<double> SoftMaxDx(const std::vector<double>& softmax, const std::vector<double>& expected) {
    std::vector<double> vec;
    vec.resize(softmax.size());

    for (int i = 0; i < vec.size(); ++i) {
        vec[i] = softmax[i] - expected[i];
    }

    return vec;
}
