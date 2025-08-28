cmake_minimum_required(VERSION 3.10)
project(OpenCVExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

add_executable(opencv_demo main.cpp)

install(TARGETS opencv_demo DESTINATION bin)

target_link_libraries(opencv_demo ${OpenCV_LIBS} ${CURL_LIBRARIES})
target_include_directories(opencv_demo PRIVATE ${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS})