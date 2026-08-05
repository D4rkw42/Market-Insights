// Definição de uma função de erro

#pragma once

#include <functional>
#include <vector>
#include <map>

// Fórmula matemática genérica
using ErrFunc = std::function<double(const std::vector<double>&, const std::vector<double>&)>;

// Fórmula matemática genérica da derivada
using ErrFuncDx = std::function<double(double, double)>;

// Definição de função de ativação
class ErrorFunction {
    public:
        const ErrFunc function; // Fórmula matemática da função de ativação
        const ErrFuncDx derivative; // Derivada da função de ativação

        ErrorFunction(const ErrFunc& function, const ErrFuncDx& derivative)
            : function(function), derivative(derivative) {}

        ErrorFunction(void) = default;
        ~ErrorFunction() = default;

        inline double CalculateFromLaw(const std::vector<double>& output, const std::vector<double>& expected) const {
            return this->function(output, expected);
        }

        inline double CalculateFromDerivative(double output, double expected) const {
            return this->derivative(output, expected);
        }
};

// Lista de funções de ativação
using ErrorFunctionList = std::map<const char*, const ErrorFunction>;

// Utils

inline const ErrorFunction CreateErrorFunction(const ErrFunc& function, const ErrFuncDx& derivative) noexcept {
    return ErrorFunction(function, derivative);
}

inline const ErrorFunction& GetErrorFunction(ErrorFunctionList& list, const char* function) {
    return list[function];
}
