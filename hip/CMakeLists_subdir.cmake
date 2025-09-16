
# 获取子目录名（例如 HostVector, DeviceVector, Solver 等）
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# 拼接目标名，比如 HostVector_objs
set(TARGET_NAME "${DIR_NAME}_objs")

# 收集所有源文件（等价于 wildcard）
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)


add_library(${TARGET_NAME} OBJECT ${SRC_FILES})

# 编译选项
target_compile_options(${TARGET_NAME} PRIVATE
    -fPIC -fopenmp -O3 -std=c++14
)