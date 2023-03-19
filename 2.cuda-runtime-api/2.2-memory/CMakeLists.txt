cmake_minimum_required(VERSION 3.0)
project(runtime-api CXX C)
set(APP_NAME memory)

# 添加目录中所有源文件
# aux_source_directory(${PROJECT_SOURCE_DIR}/infer/cpugpu   INFER_SRC_FILES)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/*.cpp)
cuda_add_executable(${APP_NAME} ${SRCS})
target_link_libraries(${APP_NAME} ${LINK_LIBS})
# message("link libs ${LINK_LIBS}")
install(TARGETS ${APP_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)