cmake_minimum_required(VERSION 3.10)
project(TensorrtTutorial CXX C)

# executable file output path
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin" CACHE PATH "output directory")
# library file output path
set(LIBRARY_OUTPUT_PATH    "${CMAKE_BINARY_DIR}/lib" CACHE PATH "library output directory")
# cmake install path
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install")
# runtime 搜索路径
set(CMAKE_INSTALL_RPATH ".:lib:build/install/lib")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(XF_COMPILE_FLAGS "-g -O2 -ffunction-sections -fdata-sections -funwind-tables")
set(XF_COMPILE_FLAGS "${XF_COMPILE_FLAGS} -fstack-protector-strong")

set(CMAKE_C_FLAGS    "${XF_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS  "${XF_COMPILE_FLAGS}")
link_directories("/usr/local/lib")

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA library status:")
    message(STATUS "    version: ${CUDA_VERSION}")
    message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "    lib path: ${CUDA_LIBRARIES}")
    message(STATUS "    install path: ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc) 
    include_directories(${CUDA_INCLUDE_DIRS})
    set(LINK_LIBS ${LINK_LIBS} ${CUDA_LIBRARIES} -lcuda)
else()
    message(FATA_ERROR "Do not find CUDA")
endif()

add_subdirectory(1.1-cuinit)
add_subdirectory(1.2-context)
add_subdirectory(1.3-memeory)

