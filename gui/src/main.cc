#include <iostream>
#include <QApplication>

#include "MainWindow.h"

extern "C" int cpp_main(int argc, char** argv);

int cpp_main(int argc, char** argv) {
  std::cout << "Hi from C++" << std::endl;

  QApplication a(argc, argv);

  MainWindow window;
  window.show();

  return a.exec();
}
