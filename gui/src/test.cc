#include "ffi.h"

extern "C" {
  int cpp_main(int, char**);
}

void neural_network_destroy(FFINeuralNetwork*) {}

FFINeuralNetwork* neural_network_create(
    size_t, size_t, size_t, size_t, float, float) {
  return nullptr;
}

extern "C" ssize_t neural_network_feed(
    FFINeuralNetwork* nn,
    const float* input) {
  return -1;
}

extern "C" ssize_t neural_network_train_one(
    FFINeuralNetwork* nn,
    size_t label,
    const float* input) {
  return -1;
}

int main(int argc, char** argv) {
  return cpp_main(argc, argv);
}
