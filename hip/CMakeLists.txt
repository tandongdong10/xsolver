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

# object file
add_library(nompikernel OBJECT mpicpu.cpp)
add_library(hipsolver-C OBJECT xsolver_c.cpp)
add_library(xsolver-C SHARED
    $<TARGET_OBJECTS:hipsolver-C>
    $<TARGET_OBJECTS:nompikernel>
)

if(HAVE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        message("tdd")
        target_compile_definitions(nompikernel PRIVATE -DHAVE_MPI)
        target_compile_definitions(hipsolver-C PRIVATE -DHAVE_MPI)
    else()
        message(FATAL_ERROR "MPI requested but not found!")
    endif()
endif()


set_target_properties(nompikernel PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
set_target_properties(hipsolver-C  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)


target_compile_options(nompikernel PRIVATE
    -std=c++14 -fopenmp -O3 
)
target_compile_options(hipsolver-C PRIVATE
    -fopenmp -O3 -std=c++14
)

# add_subdirectory(HostVector)
# add_subdirectory(HostMatrix)
# add_subdirectory(Precond)
# add_subdirectory(Solver)




# --------------------------
# install rule
# --------------------------

# 1. install share library
install(TARGETS xsolver-C
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# 2. install the header file

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)