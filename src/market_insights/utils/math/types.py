# Clase que representa uma notação científica

class ScientificNotation:
    def __init__(self, value: float):
        # Tranformação explícita em float
        value = float(value)

        # Transformação em notação científica

        notation = f"{value:e}"
        splitted_n = notation.split("e")

        # Obtém a mantissa e expoente

        self.mantissa = float(splitted_n[0])
        self.expoent = float(splitted_n[1])

    # Obtém o valor representado pela notação científica

    def get_value(self):
        return self.mantissa * pow(10, self.expoent)
