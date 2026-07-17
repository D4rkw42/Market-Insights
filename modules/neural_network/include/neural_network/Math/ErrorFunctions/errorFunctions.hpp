// Funções de cálculo de erro

#pragma once

#include <cmath>
#include <vector>

// OBS: as derivadas retornam deltas!!!!

// Média do Erro Quadrático

inline double MeanSquaredError(const std::vector<double>& output, const std::vector<double>& expected) noexcept {
    double L = 0;

    for (int i = 0; i < output.size(); ++i) {
        double v = expected[i] - output[i];
        L += v * v;
    }

    return L / output.size();
}

inline double MeanSquaredErrorDx(double output, double expected) noexcept {
    return expected - output;
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

inline double MeanAbsoluteErrorDx(double output, double expected) {
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

// Focal Loss

inline double FocalLoss(double Pt, double alpha, double gama) noexcept {
    return -alpha * std::pow(1 - Pt, gama) * std::log10(Pt);
}

inline double FocalLossDx(double Pt, double alpha, double gama) noexcept {
    return alpha * gama * std::pow(1 - Pt, gama - 1) * std::log(Pt) - alpha * std::pow(1 - Pt, gama) / Pt;
}
