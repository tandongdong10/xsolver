# Set compilation options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

# set object target
add_library(mpikernel_obj OBJECT mpicpu.cpp)
add_library(mpicudasolver_obj OBJECT mpicudasolver-C-v1.cu)

# -fPIC
set_target_properties(mpikernel_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(mpicudasolver_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Generate shared library
add_library(xsolver-C SHARED
    $<TARGET_OBJECTS:mpikernel_obj>
    $<TARGET_OBJECTS:mpicudasolver_obj>
)


# find MKL
find_package(MKL REQUIRED)
# include_directories(${MKL_ROOT}/include)

if(HAVE_MPI)
    find_package(MPI REQUIRED)
    if(MPI_FOUND)
        target_compile_definitions(mpikernel_obj PRIVATE -DHAVE_MPI)
        target_compile_definitions(mpicudasolver_obj PRIVATE -DHAVE_MPI)
        set(CMAKE_C_COMPILER ${MPI_C_COMPILER} CACHE STRING "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER} CACHE STRING "C++ compiler" FORCE)
    else()
        message(FATAL_ERROR "MPI requested but not found!")
    endif()
endif()



# link MKL、pthread、math library
target_link_libraries(xsolver-C PRIVATE
    MKL::MKL
    pthread
    m
)


# set output library name
set_target_properties(xsolver-C PROPERTIES
    OUTPUT_NAME "xsolver-C"
)


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
file(GLOB MY_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${MY_HEADERS} DESTINATION include)

# 3.install the extern library bolsig_x86_64.so
# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/bolsig_x86_64.so
#         DESTINATION lib)