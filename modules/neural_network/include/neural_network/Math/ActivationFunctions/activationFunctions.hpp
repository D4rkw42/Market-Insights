// Funções de ativação

#pragma once

#include <cmath>
#include <vector>

#include <neural_network/Math/math.hpp>

// Sigmoid

inline double Sigmoid(double x) noexcept {
    return 1.0 / (1 + std::exp(-x));
}

inline double SigmoidDx(double x) noexcept {
    double sigX = Sigmoid(x);
    return sigX * (1 - sigX);
}

// Tangente Hiperbólica

inline double Tanh(double x) noexcept {
    return std::tanh(x);
}

inline double TanhDx(double x) noexcept {
    double tanhX = Tanh(x);
    return 1 - tanhX * tanhX;
}

// Função Identidade

constexpr double Identity(double x) noexcept {
    return x;
}

constexpr double IdentityDx(double x) noexcept {
    return 1;
}

// LeakyReLu

inline double LeakyReLu100(double x) noexcept {
    return (x > 0)? x : x / 100;
}

inline double LeakyReLu100Dx(double x) noexcept {
    return (x > 0)? 1 : 0.01f;
}

inline double LeakyReLu1000(double x) noexcept {
    return (x > 0)? x : x / 1000;
}

inline double LeakyReLu1000Dx(double x) noexcept {
    return (x > 0)? 1 : 0.001f;
}

// Gaussian Error Linear Unit

inline double GELU(double x) noexcept {
    static const double a = std::sqrt(2.0 / PI);
    return (x * 0.5f) * (1 + Tanh(a * (x + 0.044715 * std::pow(x, 3))));
}

inline double GELUDx(double x) noexcept {
    static const double a = std::sqrt(2.0 / PI);
    const double u = a * (x + 0.044715 * std::pow(x, 3));

    const double TanhU = Tanh(u);

    return 0.5 * (1 + TanhU) + (x * 0.5f) * (1 - TanhU * TanhU) * a * (1 + 0.134145 * x * x);
}

// Softmax

inline std::vector<double> SoftMax(const std::vector<double>& logits) noexcept {
    std::vector<double> vec = logits;

    double sum = 0;

    for (double l : logits) {
        sum += std::exp(l);
    }

    for (int i = 0; i < vec.size(); ++i) {
        vec[i] = std::exp(vec[i]) / sum;
    }

    return vec;
}

inline std::vector<double> SoftMaxDx(const std::vector<double>& softmax, const std::vector<double>& expected) noexcept {
    std::vector<double> vec;
    vec.resize(softmax.size());

    for (int i = 0; i < vec.size(); ++i) {
        vec[i] = softmax[i] - expected[i];
    }

    return vec;
}
