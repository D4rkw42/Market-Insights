#include <iostream>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void HelloWorld(void)
{
    std::cout << "Hello World!\n";
}

PYBIND11_MODULE(neural_network, m)
{
    m.def("hello_world", &HelloWorld);
}
