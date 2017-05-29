#include "MainWindow.h"
#include "NeuralNetworkInputWidget.h"

#include <QGridLayout>
#include <QSlider>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QIntValidator>

MainWindow::MainWindow(QWidget *parent)
  : QMainWindow(parent)
{
  setWindowTitle("Multi-layer neural network");
  setCentralWidget(new QWidget());

  auto* layout = new QGridLayout();

  {
    m_input = new NeuralNetworkInputWidget();
    layout->addWidget(m_input, 0, 0, 1, 4, Qt::AlignCenter);
  }

  {
    auto* clearButton = new QPushButton();
    clearButton->setText("Clear");
    connect(clearButton, SIGNAL(clicked()), this, SLOT(clearInput()));

    layout->addWidget(clearButton, 1, 0, 1, 4, Qt::AlignCenter);
  }

  {
    auto* label = new QLabel();
    label->setText("Learning rate");
    layout->addWidget(label, 2, 0, Qt::AlignTop);

    m_learningRateInput = new QSlider(Qt::Horizontal);
    m_learningRateInput->setMinimum(0);
    m_learningRateInput->setMaximum(100);
    layout->addWidget(m_learningRateInput, 3, 0, Qt::AlignTop);
  }

  {
    auto* label = new QLabel();
    label->setText("Momentum rate");
    layout->addWidget(label, 2, 1, Qt::AlignTop);

    m_momentumRateInput = new QSlider(Qt::Horizontal);
    m_momentumRateInput->setMinimum(0);
    m_momentumRateInput->setMaximum(100);
    layout->addWidget(m_momentumRateInput, 3, 1, Qt::AlignTop);
  }

  {
    auto* label = new QLabel();
    label->setText("Hidden layer count");
    layout->addWidget(label, 2, 2, Qt::AlignTop);

    m_hiddenLayerCountInput = new QLineEdit();
    auto* validator = new QIntValidator();
    validator->setBottom(0);
    m_hiddenLayerCountInput->setValidator(validator);
    layout->addWidget(m_hiddenLayerCountInput, 3, 2, Qt::AlignTop);
  }

  {
    auto* label = new QLabel();
    label->setText("Hidden neuron count");
    layout->addWidget(label, 2, 3, Qt::AlignTop);

    m_hiddenNeuronCountInput = new QLineEdit();
    auto* validator = new QIntValidator();
    validator->setBottom(0);
    m_hiddenNeuronCountInput->setValidator(validator);
    layout->addWidget(m_hiddenNeuronCountInput, 3, 3, Qt::AlignTop);
  }

  {
    m_trainingValue = new QLineEdit();
    m_trainingValue->setPlaceholderText("Training label");
    auto* validator = new QIntValidator();
    validator->setBottom(0);
    validator->setTop(9);
    m_trainingValue->setValidator(validator);
    layout->addWidget(m_trainingValue, 4, 0, Qt::AlignLeft);

    auto* trainButton = new QPushButton();
    trainButton->setText("Train");
    layout->addWidget(trainButton, 4, 1, 1, 2, Qt::AlignCenter);
    connect(trainButton, SIGNAL(clicked()), this, SLOT(trainNetwork()));

    m_trainingResult = new QLineEdit();
    m_trainingResult->setPlaceholderText("Result");
    m_trainingResult->setReadOnly(true);
    layout->addWidget(m_trainingResult, 4, 3, Qt::AlignLeft);
  }

  {
    auto* feedButton = new QPushButton();
    feedButton->setText("Feed");
    layout->addWidget(feedButton, 5, 1, 1, 2, Qt::AlignCenter);
    connect(feedButton, SIGNAL(clicked()), this, SLOT(feedNetwork()));

    m_feedingResult = new QLineEdit();
    m_feedingResult->setPlaceholderText("Result");
    m_feedingResult->setReadOnly(true);
    layout->addWidget(m_feedingResult, 5, 3, Qt::AlignLeft);
  }

  centralWidget()->setLayout(layout);
}

bool MainWindow::updateState() {
  {
    auto hiddenLayerCountText = m_hiddenLayerCountInput->text();
    if (hiddenLayerCountText.isEmpty())
      return false;

    bool ok = false;
    auto hiddenLayerCount = hiddenLayerCountText.toInt(&ok, 10);
    if (!ok)
      return false;

    m_state.setHiddenLayerCount(hiddenLayerCount);
  }

  {
    auto hiddenNeuronCountText = m_hiddenNeuronCountInput->text();
    if (hiddenNeuronCountText.isEmpty())
      return false;

    bool ok = false;
    auto hiddenNeuronCountPerLayer = hiddenNeuronCountText.toInt(&ok, 10);
    if (!ok)
      return false;

    m_state.setHiddenNeuronCountPerLayer(hiddenNeuronCountPerLayer);
  }

  int momentumRate = m_momentumRateInput->value();
  m_state.setMomentumRate(static_cast<float>(momentumRate) / 100.0);

  int learningRate = m_learningRateInput->value();
  m_state.setLearningFactor(static_cast<float>(learningRate) / 100.0);

  return true;
}

void MainWindow::clearInput() {
  m_input->clear();
}

void MainWindow::trainNetwork() {
  if (!trainNetworkInternal())
    m_trainingResult->setText("<error>");
}
bool MainWindow::trainNetworkInternal() {
  updateState();

  auto labelText = m_trainingValue->text();
  if (labelText.isEmpty())
    return false;
  bool ok = false;
  auto label = labelText.toUInt(&ok, 10);
  if (!ok)
    return false;

  auto result = m_state.train(m_input->inputData(), label);
  if (result < 0)
    return false;

  m_trainingResult->setText(QString::number(result, 10));
  return true;
}

void MainWindow::feedNetwork() {
  updateState();
  auto result = m_state.feed(m_input->inputData());
  if (result < 0)
    m_feedingResult->setText("<error>");
  else
    m_feedingResult->setText(QString::number(result, 10));
}

MainWindow::~MainWindow() {
}
