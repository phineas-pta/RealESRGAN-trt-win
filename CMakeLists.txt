cmake_minimum_required(VERSION 2.6)

project(real-esrgan)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
set(OpenCV_DIR ███/OpenCV/build)
set(CMAKE_CUDA_ARCHITECTURES ███)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -W4 -D_MWAITXINTRIN_H_INCLUDED")

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV)

enable_language(CUDA)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(real-esrgan real-esrgan.cpp preprocess.cu postprocess.cu)

target_link_libraries(real-esrgan nvinfer nvinfer_plugin)
target_link_libraries(real-esrgan cudart)
target_link_libraries(real-esrgan ${OpenCV_LIBS})
target_link_libraries(real-esrgan Threads::Threads)
