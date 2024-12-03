cmake_minimum_required(VERSION 3.10)
link_directories(C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build)
set(CMAKE_PREFIX_PATH "C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build/install")

project(NV12Converter)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(nv12_converter src/main.cpp src/rgb2nv12.cpp src/nv122rgb.cpp)
target_link_libraries(nv12_converter ${OpenCV_LIBS})


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
