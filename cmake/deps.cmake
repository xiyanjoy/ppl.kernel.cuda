if(NOT HPCC_DEPS_DIR)
    set(HPCC_DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps)
endif()

# forces to install libraries to `lib`, not `lib64` or others
set(CMAKE_INSTALL_LIBDIR lib)

# --------------------------------------------------------------------------- #

if(CMAKE_COMPILER_IS_GNUCC)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9.0)
        message(FATAL_ERROR "gcc >= 4.9.0 is required.")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0.0)
        message(FATAL_ERROR "clang >= 6.0.0 is required.")
    endif()
endif()

# --------------------------------------------------------------------------- #

if(APPLE)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/opt/libomp/include -Wno-unused-command-line-argument")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/opt/libomp/include -Wno-unused-command-line-argument")
    endif()
endif()

# --------------------------------------------------------------------------- #

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${HPCC_DEPS_DIR})
set(FETCHCONTENT_QUIET OFF)

if(PPLNN_HOLD_DEPS)
    set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
endif()

# --------------------------------------------------------------------------- #

find_package(Git QUIET)
if(NOT Git_FOUND)
    message(FATAL_ERROR "git is required.")
endif()

if(NOT PPLNN_DEP_HPCC_VERSION)
    set(PPLNN_DEP_HPCC_VERSION master)
endif()

if(PPLNN_DEP_HPCC_PKG)
    FetchContent_Declare(hpcc
        URL ${PPLNN_DEP_HPCC_PKG}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
else()
    if(NOT PPLNN_DEP_HPCC_GIT)
        set(PPLNN_DEP_HPCC_GIT "https://github.com/openppl-public/hpcc.git")
    endif()
    FetchContent_Declare(hpcc
        GIT_REPOSITORY ${PPLNN_DEP_HPCC_GIT}
        GIT_TAG ${PPLNN_DEP_HPCC_VERSION}
        SOURCE_DIR ${HPCC_DEPS_DIR}/hpcc
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/hpcc-build
        SUBBUILD_DIR ${HPCC_DEPS_DIR}/hpcc-subbuild)
endif()

FetchContent_GetProperties(hpcc)
if(NOT hpcc_POPULATED)
    FetchContent_Populate(hpcc)
    include(${hpcc_SOURCE_DIR}/cmake/hpcc-common.cmake)
endif()

# --------------------------------------------------------------------------- #

if(PPLNN_HOLD_DEPS)
    set(PPLCOMMON_HOLD_DEPS ON)
endif()

if(NOT PPLNN_DEP_PPLCOMMON_VERSION)
    set(PPLNN_DEP_PPLCOMMON_VERSION master)
endif()

if(PPLNN_USE_X86_64)
    set(PPLCOMMON_USE_X86_64 ON)
endif()
if(PPLNN_USE_AARCH64)
    set(PPLCOMMON_USE_AARCH64 ON)
endif()
if(PPLNN_USE_CUDA)
    set(PPLCOMMON_USE_CUDA ON)
endif()

if(PPLNN_DEP_PPLCOMMON_PKG)
    hpcc_declare_pkg_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_PKG})
else()
    if(NOT PPLNN_DEP_PPLCOMMON_GIT)
        set(PPLNN_DEP_PPLCOMMON_GIT "https://github.com/openppl-public/ppl.common.git")
    endif()
    hpcc_declare_git_dep(pplcommon
        ${PPLNN_DEP_PPLCOMMON_GIT}
        ${PPLNN_DEP_PPLCOMMON_VERSION})
endif()

set(CUTLASS_GIT_SRC "git@gitlab.sz.sensetime.com:HPC/cutlass.git")
set(CUTLASS_GIT_VERSION "9d8d9b5160ffe509ec6611d7f968a2643a7f65ba")
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0.0"
    AND CUDA_VERSION VERSION_GREATER_EQUAL "11.4")
    set(CUTLASS_GIT_SRC "https://github.com/NVIDIA/cutlass.git")
    set(CUTLASS_GIT_VERSION "39c6a83f231d6db2bc6b9c251e7add77d68cbfb4") #for flash attn2
endif()
hpcc_declare_git_dep(cutlass
    ${CUTLASS_GIT_SRC}
    ${CUTLASS_GIT_VERSION})