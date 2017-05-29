#include "NeuralNetworkInputWidget.h"
#include <QPainter>
#include <QMouseEvent>
#include <QPaintEvent>

const size_t SIZE_RATIO = 16;

NeuralNetworkInputWidget::NeuralNetworkInputWidget(QWidget* parent)
  : QWidget(parent)
{
  setFixedSize(NEURAL_NETWORK_INPUT_SIZE * SIZE_RATIO, NEURAL_NETWORK_INPUT_SIZE * SIZE_RATIO);
}

NeuralNetworkInputWidget::~NeuralNetworkInputWidget() {}

void NeuralNetworkInputWidget::mousePressEvent(QMouseEvent*) {
  m_userIsDrawing = true;
}

void NeuralNetworkInputWidget::mouseMoveEvent(QMouseEvent* e) {
  const float STEP = 0.05;

  if (!m_userIsDrawing)
    return;

  QPoint point = e->pos();
  point /= static_cast<float>(SIZE_RATIO);

  if (point.x() < 0 || point.x() > NEURAL_NETWORK_INPUT_SIZE)
    return; // How could this happen?

  if (point.y() < 0 || point.y() > NEURAL_NETWORK_INPUT_SIZE)
    return;

  float& p = m_pixels[point.x()][point.y()];
  p = std::min(p + STEP, 1.0f);

  update(point.x() * SIZE_RATIO, point.y() * SIZE_RATIO, SIZE_RATIO, SIZE_RATIO);
}

void NeuralNetworkInputWidget::mouseReleaseEvent(QMouseEvent*) {
  m_userIsDrawing = false;
}

void NeuralNetworkInputWidget::paintEvent(QPaintEvent* e) {
  QPainter painter(this);
  for (const auto& region : e->region()) {
    // We draw the background before-hand as white to optimize large repaints,
    // in which most of the pixels are white and we can skip painting point by
    // point.
    //
    // We could keep a pixmap instead of this, but seems somewhat unnecessary.
    painter.setBrush(QBrush(QColor(255, 255, 255)));
    painter.setPen(Qt::PenStyle::NoPen);
    painter.drawRect(region);

    QPoint topLeft = region.topLeft();
    topLeft /= SIZE_RATIO;

    QPoint bottomRight = region.bottomRight();
    bottomRight /= SIZE_RATIO;

    for (size_t x = topLeft.x(); x < bottomRight.x(); ++x) {
      for (size_t y = topLeft.y(); y < bottomRight.y(); ++y) {
        if (m_pixels[x][y] == 0.0)
          continue;
        int ratio = (1.0 - m_pixels[x][y]) * 255.0;
        painter.setBrush(QBrush(QColor(ratio, ratio, ratio)));
        painter.drawRect(
            x * SIZE_RATIO,
            y * SIZE_RATIO,
            SIZE_RATIO,
            SIZE_RATIO);
      }
    }
  }
}

void NeuralNetworkInputWidget::clear() {
  memset(m_pixels, 0, sizeof(m_pixels));
  update(visibleRegion());
}
