#include "neural_network/LSTM/LSTM.hpp"

LSTM::LSTM(void) {
    this->num = 10;
}
 
void LSTM::SetNum(int n) {
    this->num = n;
}
