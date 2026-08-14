# Rede Neural para Predição de Movimento de Preço de Ativos

Considerando N = x como o número de entradas e O = 3 como o número de saídas:

OBS: O mínimo de neurônios em cada camada oculta é O * 3

N       (BASIC) -> Identity
N * 2   (LSTM)  -> Tanh
N       (LSTM)  -> Tanh
N/2     (LSTM)  -> Tanh
N/4     (LSTM)  -> Tanh
N/8     (LSTM)  -> GELU
O       (BASIC) -> Identity

SOFTMAX
