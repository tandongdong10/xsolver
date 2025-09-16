# find hip environment
find_package(HIP REQUIRED)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
    message(STATUS "HIP_DIR = ${HIP_DIR}")
    message(STATUS "HIP_INCLUDE_DIRS = ${HIP_INCLUDE_DIRS}")
    message(STATUS "HIP_LIBRARY_DIRS = ${HIP_LIBRARY_DIRS}")
    message(STATUS "HIP_LIBRARIES = ${HIP_LIBRARIES}")
    message(STATUS "HIP_EXECUTABLE = ${HIP_EXECUTABLE}")
    message(STATUS "HIP_HIPCC_EXECUTABLE = ${HIP_HIPCC_EXECUTABLE}")
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE} CACHE STRING "C++ compiler" FORCE)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated>)
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in ${ROCM_PATH}/hip or the variable HIP_PATH is set to point to the right location.")
endif()




if(HAVE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message("use -DHAVE_MPI")
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-DHAVE_MPI>)
    else()
        message(FATAL_ERROR "MPI requested but not found!")
    endif()
endif()


add_subdirectory(HostVector)
add_subdirectory(HostMatrix)
add_subdirectory(Precond)
add_subdirectory(Solver)

# object file
add_library(mpicpu_obj OBJECT mpicpu.cpp)
add_library(xsolver_c_obj OBJECT xsolver_c.cpp)
add_library(xsolver-C SHARED
    $<TARGET_OBJECTS:xsolver_c_obj>
    $<TARGET_OBJECTS:mpicpu_obj>
    $<TARGET_OBJECTS:HostVector_objs>
    $<TARGET_OBJECTS:HostMatrix_objs>
    $<TARGET_OBJECTS:Precond_objs>
    $<TARGET_OBJECTS:Solver_objs>
)


set_target_properties(mpicpu_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
set_target_properties(xsolver_c_obj  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)


target_compile_options(mpicpu_obj PRIVATE
    -std=c++14 -fopenmp -O3 
)
target_compile_options(xsolver_c_obj PRIVATE
    -fopenmp -O3 -std=c++14
)



# Test libxsolver-C.so


if(HAVE_MPI)
    add_executable(main_solve_mpi main_solve_mpi.cpp)

    target_include_directories(main_solve_mpi PRIVATE 
        ${CMAKE_SOURCE_DIR}
    )

    target_compile_options(main_solve_mpi PRIVATE
        -std=c++14 -Wno-deprecated -O3
    )

    target_link_libraries(main_solve_mpi PRIVATE
        xsolver-C
        mkl_intel_lp64
        mkl_intel_thread
        mkl_core
        iomp5
        rocblas
        hipsparse
        hipblas
        pthread
        mpi
    )
else()
    add_executable(main_solve_base main_solve_base.cpp)

    target_include_directories(main_solve_base PRIVATE 
        ${CMAKE_SOURCE_DIR}
    )

    target_compile_options(main_solve_base PRIVATE
        -std=c++14 -Wno-deprecated -O3
    )

    target_link_libraries(main_solve_base PRIVATE
        xsolver-C
        mkl_intel_lp64
        mkl_intel_thread
        mkl_core
        iomp5
        rocblas
        hipsparse
        hipblas
        pthread
    )
endif()



# --------------------------
# install rule
# --------------------------

# 1. install share library
install(TARGETS xsolver-C
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)


if(HAVE_MPI)
    install(TARGETS main_solve_mpi
            RUNTIME DESTINATION bin
    )
else()
    install(TARGETS main_solve_base
            RUNTIME DESTINATION bin
    )
endif()

# 2. install the header file

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)