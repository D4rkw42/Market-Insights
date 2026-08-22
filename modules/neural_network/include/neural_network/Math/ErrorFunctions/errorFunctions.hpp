// Funções de cálculo de erro

#pragma once

#include <cmath>
#include <vector>

// Média do Erro Quadrático

inline double MeanSquaredError(const std::vector<double>& output, const std::vector<double>& expected) noexcept {
    double L = 0;

    for (int i = 0; i < output.size(); ++i) {
        double v = expected[i] - output[i];
        L += v * v;
    }

    return L / output.size();
}

inline double SquaredErrorDx(double output, double expected) noexcept {
    return -2 * (expected - output);
}

// Média do Erro Absoluto

inline double MeanAbsoluteError(const std::vector<double>& output, const std::vector<double>& expected) noexcept {
    double L = 0;

    for (int i = 0; i < output.size(); ++i) {
        double v = expected[i] - output[i];
        L += std::abs(v);
    }

    return L / output.size();
}

inline double AbsoluteErrorDx(double output, double expected) {
    if (expected == output) {
        return 0;
    }

    return (expected > output)? 1 : -1;
}

// Cross Entropy

inline double CrossEntropy(const std::vector<double>& output, const std::vector<double>& expected) noexcept {
    double L = 0;

    for (int i = 0; i < output.size(); ++i) {
        L += expected[i] * std::log10(output[i]);
    }

    return -L;
}

inline double CrossEntropyDx(double output, double expected) noexcept {
    return -expected/output;
}
