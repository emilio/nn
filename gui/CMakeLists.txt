cmake_minimum_required(VERSION 2.8.5)
project(neural-networks-gui CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

if(NOT CMAKE_CONFIGURATION_TYPES)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Whether to build in `Debug` or `Release` mode." FORCE)
  endif()
endif()

include_directories(dir)
find_package(Qt5Widgets)

add_library(neural-networks-gui
  src/main.cc
  src/MainWindow.cc
  src/NeuralNetworkInputWidget.cc
  src/NeuralNetworkState.cc
)

target_link_libraries(neural-networks-gui Qt5::Widgets)

add_executable(test
  src/test.cc
)
target_link_libraries(test neural-networks-gui)

install_targets(. neural-networks-gui neural-networks-gui)
