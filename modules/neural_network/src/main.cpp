#include <memory>

#include <pybind11/pybind11.h>

#include "neural_network/LSTM/LSTM.hpp"

namespace py = pybind11;

/* Definições gerais */

#define MODULE_NAME neural_network
#define MODULE_GIL_MODE py::mod_gil_not_used()

/* Definição da biblioteca */

PYBIND11_MODULE(MODULE_NAME, m, MODULE_GIL_MODE) {
    
    /* Exportando a classe LSTM, que representa a rede neural */

    py::class_<LSTM, std::shared_ptr<LSTM>>(m, "LSTM")
        .def(py::init<>())
        .def_readwrite("num", &LSTM::num)
        .def("set_num", &LSTM::SetNum);
}
