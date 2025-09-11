cmake_minimum_required(VERSION 3.18)
project(xsolver LANGUAGES C CXX)

# choose backend (default INTEL )
set(XSOLVER_BACKEND "INTEL" CACHE STRING "Backend for xsolver: INTEL | CUDA | HIP")
set_property(CACHE XSOLVER_BACKEND PROPERTY STRINGS INTEL CUDA HIP)

# Whether to enable the MPI
option(HAVE_MPI "Enable MPI support" OFF)


# Select subdirectory based on backend
if(XSOLVER_BACKEND STREQUAL "INTEL")
    add_subdirectory(intel)
elseif(XSOLVER_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    add_subdirectory(cuda)
elseif(XSOLVER_BACKEND STREQUAL "HIP")
    find_package(HIP REQUIRED)
    add_subdirectory(hip)
else()
    message(FATAL_ERROR "Unsupported XSOLVER_BACKEND: ${XSOLVER_BACKEND}")
endif()
