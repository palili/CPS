cmake_minimum_required(VERSION 3.16)
project(CPS VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CPS_BUILD_TESTS "Build CPS unit tests" ON)

add_library(cps_core STATIC
    src/clustering.cpp
)

target_include_directories(cps_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_compile_definitions(cps_core PUBLIC
    FULL_COVER_DMC_COEFFICIENT=1
)

if(MSVC)
    target_compile_options(cps_core PUBLIC /W4 /EHsc $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(cps_core PUBLIC -Wall -Wextra $<$<CONFIG:Release>:-O3>)
endif()

add_executable(cps_cli
    src/cps_main.cpp
)
target_link_libraries(cps_cli PRIVATE cps_core)

if(CPS_BUILD_TESTS)
    enable_testing()
    add_executable(slot_and_icc_tests
        tests/slot_and_icc_tests.cpp
    )
    target_link_libraries(slot_and_icc_tests PRIVATE cps_core)
    add_test(NAME slot_and_icc_tests COMMAND slot_and_icc_tests)
endif()
