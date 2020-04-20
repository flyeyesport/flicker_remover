cmake_minimum_required(VERSION 3.3)
project(flicker_remover)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(NAMES
        main.cxx
        flicker_remover.hpp
        flicker_remover_cpu.hpp
        circular_buffer.hpp
        open_cl_kernels.hpp
        flicker_remover.cxx
        flicker_remover_cpu.cxx
        open_cl_kernels.cxx
        boolean_array_2_d.cxx
        boolean_array_2_d.hpp
        )

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(flicker_remover ${NAMES})
target_link_libraries(flicker_remover ${OpenCV_LIBRARIES})
