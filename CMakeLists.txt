cmake_minimum_required(VERSION 2.8.3)
set(OpenCV_DIR "/home/rjp/MyOpenCV/opencv-3.4-install/share/OpenCV")
project(test4opencv_24)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(test_fr
  main.cpp
)

target_link_libraries(test_fr
  ${OpenCV_LIBS}
)
