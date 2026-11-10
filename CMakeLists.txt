cmake_minimum_required(VERSION 3.18)
project(TurboReg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_DEMO_CPP "Build Sim3 command line tool" ON)

# LibTorch
find_package(Torch REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# TurboReg Core Library
include_directories(turboreg/include)
add_subdirectory(turboreg)

# Demos
if (BUILD_DEMO_CPP)
    add_executable(turboreg_sim3 apps/sim3_cli.cpp)
    target_link_libraries(turboreg_sim3 PRIVATE turboreg)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_sim3 tests/test_sim3.cpp)
    target_link_libraries(test_sim3 PRIVATE turboreg)
    add_test(NAME sim3_contract COMMAND test_sim3)
endif()
