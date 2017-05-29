#pragma once

#include <unistd.h>

struct FFINeuralNetwork;
extern "C" void neural_network_destroy(FFINeuralNetwork*);

extern "C" FFINeuralNetwork* neural_network_create(
    size_t input_count,
    size_t output_count,
    size_t hidden_layer_count,
    size_t hidden_neuron_count_per_layer,
    float learning_factor,
    float momentum_rate);

extern "C" ssize_t neural_network_feed(
    FFINeuralNetwork* nn,
    const float* input);

extern "C" ssize_t neural_network_train_one(
    FFINeuralNetwork* nn,
    size_t label,
    const float* input);
