cmake_minimum_required(VERSION 3.10)
link_directories(C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build)
set(CMAKE_PREFIX_PATH "C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build/install")

project(QRCodeReader)


set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(QRCodeReader src/main.cpp)


target_link_libraries(QRCodeReader ${OpenCV_LIBS})


include_directories(${OpenCV_INCLUDE_DIRS})
