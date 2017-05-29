#pragma once

#include <QMainWindow>
#include "NeuralNetworkState.h"

class NeuralNetworkInputWidget;
class QLineEdit;
class QSlider;

class MainWindow : public QMainWindow
{
  Q_OBJECT

  // Weak, but is held alive by the layout, which only goes away as we go away,
  // so it's fine to use it all the time.
  NeuralNetworkInputWidget* m_input { nullptr };
  QLineEdit* m_hiddenLayerCountInput { nullptr };
  QLineEdit* m_hiddenNeuronCountInput { nullptr };
  QSlider* m_learningRateInput { nullptr };
  QSlider* m_momentumRateInput { nullptr };
  QLineEdit* m_trainingValue { nullptr };
  QLineEdit* m_trainingResult { nullptr };
  QLineEdit* m_feedingResult { nullptr };

  NeuralNetworkState m_state;

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow();

 private slots:
  bool updateState();
  void clearInput();
  void feedNetwork();
  void trainNetwork();
  bool trainNetworkInternal();
};
