// Definição de uma função de ativação

#include <functional>
#include <map>

// Fórmula matemática genérica
using ActFunc = std::function<double(double)>;

// Definição de função de ativação
class ActivationFunction {
    public:
        const ActFunc function; // Fórmula matemática da função de ativação
        const ActFunc derivative; // Derivada da função de ativação

        ActivationFunction(const ActFunc& function, const ActFunc& derivative)
            : function(function), derivative(derivative) {}

        inline double CalculateFromLaw(double x) {
            return this->function(x);
        }

        inline double CalculateFromDerivative(double x) {
            return this->derivative(x);
        }
};

// Lista de funções de ativação
using ActivationFunctionList = std::map<const char*, const ActivationFunction>;

// Utils

ActivationFunction CreateActivationFunction(const ActFunc& function, const ActFunc& derivative) noexcept {
    return ActivationFunction(function, derivative);
}
