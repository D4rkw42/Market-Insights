# An application built upon a LSTM neural network that creates insights about financial market for buying or selling operations
# D4rkw42

from neural_network import LSTM

def main():
    lstm = LSTM()

    print(lstm.num)

    lstm.set_num(20)

    print(lstm.num)
