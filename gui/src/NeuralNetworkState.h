#pragma once

#include <vector>
#include "ffi.h"

const size_t NEURAL_NETWORK_INPUT_SIZE = 64;
const size_t NEURAL_NETWORK_OUTPUT_COUNT = 10; // 0..9

class NeuralNetworkState {
  FFINeuralNetwork* m_network { nullptr };

  size_t m_hiddenLayerCount { 0 };
  size_t m_hiddenNeuronCountPerLayer { 0 };
  float m_learningFactor { 0.0 };
  float m_momentumRate { 0.0 };
  bool m_dirty { true };

  void ensureUpToDate();

 public:
  ssize_t feed(const float* input);
  ssize_t train(const float* input, size_t label);

  size_t hiddenLayerCount() const { return m_hiddenLayerCount; }
  void setHiddenLayerCount(size_t);

  size_t hiddenNeuronCountPerLayer() const { return m_hiddenNeuronCountPerLayer; }
  void setHiddenNeuronCountPerLayer(size_t);

  float learningFactor() const { return m_learningFactor; }
  void setLearningFactor(float);

  float momentumRate() const { return m_momentumRate; }
  void setMomentumRate(float);

  NeuralNetworkState() {};
  ~NeuralNetworkState();
};
