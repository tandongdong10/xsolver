message(STATUS "CMAKE_HIP_COMPILER = ${CMAKE_HIP_COMPILER}")
message("${CMAKE_BINARY_DIR}")



# 收集所有源文件（等价于 wildcard）
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# 设置输出目录
# set(OBJ_DIR ${CMAKE_BINARY_DIR}/obj)

# 自定义目标，逐个编译源文件
foreach(src ${SRC_FILES})
    # 提取不带路径的文件名
    get_filename_component(fname ${src} NAME_WE)

    # 定义 object library（每个 .cpp 编译成 .o）
    add_library(${fname}_obj OBJECT ${src})

    # 设置编译器（用 HIPCC）
    set_target_properties(${fname}_obj PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    # 添加编译选项（等价于 -fPIC -fopenmp -O3 ...）
    target_compile_options(${fname}_obj PRIVATE
        -fPIC -fopenmp -O3 -std=c++14
        -UONAME -DONAME=${fname}
    )
endforeach()

