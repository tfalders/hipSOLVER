# ########################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc.
# ########################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
  set( CMAKE_Fortran_COMPILER  "gfortran" )
endif()

project( hipsolver LANGUAGES CXX Fortran )

# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )

set ( VERSION_STRING "0.1.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

if( NOT DEFINED $ENV{HIP_PATH})
    set( HIP_PATH "/opt/rocm/hip" )
else( )
    set (HIP_PATH $ENV{HIP_PATH} )
endif( )

# Append our library helper cmake path and the cmake path for hip (for convenience)
# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${HIP_PATH}/cmake )

# NOTE:  workaround until hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/llvm /opt/rocm/hip )

option( BUILD_VERBOSE "Output additional build information" OFF )

# BUILD_SHARED_LIBS is a cmake built-in; we make it an explicit option such that it shows in cmake-gui
option( BUILD_SHARED_LIBS "Build hipSOLVER as a shared library" ON )

# Quietly look for CUDA, but if not found it's not an error
# The presense of hip is not sufficient to determine if we want a rocm or cuda backend
option( USE_CUDA "Look for CUDA and use that as a backend if found" ON )
if( USE_CUDA )
    find_package( CUDA QUIET )
endif()

# Hip headers required of all clients; clients use hip to allocate device memory
if( USE_CUDA)
    find_package( HIP REQUIRED )
else( )
    find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm )
endif( )

if( USE_CUDA )
    list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
endif( )

# CMake list of machine targets
set( AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target" )

add_subdirectory( library )

include( clients/cmake/build-options.cmake )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  add_subdirectory( clients )
endif( )
