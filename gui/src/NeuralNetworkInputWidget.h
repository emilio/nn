#pragma once

#include <QWidget>
#include "NeuralNetworkState.h"

class NeuralNetworkInputWidget final : public QWidget {
  float m_pixels[NEURAL_NETWORK_INPUT_SIZE][NEURAL_NETWORK_INPUT_SIZE] { 0 };
  bool m_userIsDrawing { false };

  void paintEvent(QPaintEvent*) final;

  void mousePressEvent(QMouseEvent*) final;
  void mouseReleaseEvent(QMouseEvent*) final;
  void mouseMoveEvent(QMouseEvent*) final;

 public:
  void clear();

  const float* inputData() const { return &m_pixels[0][0]; }

  explicit NeuralNetworkInputWidget(QWidget* parent = nullptr);
  ~NeuralNetworkInputWidget();
};
