#include "NeuralNetworkState.h"

void NeuralNetworkState::ensureUpToDate() {
  if (m_network && !m_dirty)
    return;
  if (m_network)
    neural_network_destroy(m_network);
  m_network = neural_network_create(NEURAL_NETWORK_INPUT_SIZE,
                                    NEURAL_NETWORK_OUTPUT_COUNT,
                                    hiddenLayerCount(),
                                    hiddenNeuronCountPerLayer(),
                                    learningFactor(),
                                    momentumRate());
  m_dirty = false;
}

void NeuralNetworkState::setHiddenLayerCount(size_t count) {
  if (count == hiddenLayerCount())
    return;
  m_hiddenLayerCount = count;
  m_dirty = true;
}

void NeuralNetworkState::setHiddenNeuronCountPerLayer(size_t count) {
  if (count == hiddenNeuronCountPerLayer())
    return;
  m_hiddenNeuronCountPerLayer = count;
  m_dirty = true;
}

void NeuralNetworkState::setMomentumRate(float rate) {
  if (rate == momentumRate())
    return;
  m_momentumRate = rate;
  m_dirty = true;
}

void NeuralNetworkState::setLearningFactor(float factor) {
  if (factor == learningFactor())
    return;
  m_learningFactor = factor;
  m_dirty = true;
}

ssize_t NeuralNetworkState::feed(const float* input) {
  ensureUpToDate();
  return neural_network_feed(
      m_network,
      input);
}

ssize_t NeuralNetworkState::train(const float* input, size_t label) {
  ensureUpToDate();
  return neural_network_train_one(
      m_network,
      label,
      input);
}

NeuralNetworkState::~NeuralNetworkState() {
  if (m_network)
    neural_network_destroy(m_network);
}
