cmake_minimum_required(VERSION 3.1)

project(dip_week4
        LANGUAGES    CXX
        VERSION     0.0.1
)

set(CMAKE_MAKE_PROGRAM make)

set(CMAKE_CXX_COMPILER /usr/bin/c++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_MACOSX_RPATH OFF)
# set(CMAKE_THREAD_PREFER_PTHREAD ON)
# set(THREADS_PREFER_PTHREAD_FLAG ON) 

find_package(OpenCV REQUIRED)
find_package(fmt)

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

INCLUDE_DIRECTORIES( /usr/include/opencv4 )
INCLUDE_DIRECTORIES( ./inc )
INCLUDE_DIRECTORIES( /usr/include )
# INCLUDE_DIRECTORIES( /usr/include/fmt/ )

add_executable(
    dip_week4 ./src/main.cpp ./src/metric.cpp ./src/motion.cpp
)

target_link_libraries(dip_week4
    ${OpenCV_LIBS}
    fmt::fmt
    # simdjson::simdjson
)
