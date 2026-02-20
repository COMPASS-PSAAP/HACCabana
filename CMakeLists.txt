#------------------------------------------------------------------------------#
# Project settings
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.18)

project(HACCabana
  VERSION 0.1
  LANGUAGES CXX C
)

include(GNUInstallDirs)
include(FeatureSummary)
include(CMakePackageConfigHelpers)

# Where to install the package config + exported targets
set(HACCabana_INSTALL_PACKAGEDIR
    "${CMAKE_INSTALL_DATADIR}/cmake/HACCabana"
    CACHE PATH "Install location of HACCabana CMake package files"
)

# Find Kokkos
find_package(Kokkos 4 REQUIRED)

# Supported device options (optional checks)
set(HACCABANA_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)

foreach(_device IN LISTS HACCABANA_SUPPORTED_DEVICES)
  option(HACCabana_REQUIRE_${_device}
         "Require Kokkos ${_device} support"
         OFF)
  if(HACCabana_REQUIRE_${_device})
    kokkos_check(DEVICES ${_device})
  endif()
endforeach()

# Add local cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Standard dependency macro: optional unless user forces REQUIRE_*
macro(HACCabana_add_dependency)
  cmake_parse_arguments(HACCABANA_DEPENDENCY "" "PACKAGE;VERSION" "COMPONENTS" ${ARGN})

  find_package(${HACCABANA_DEPENDENCY_PACKAGE}
               ${HACCABANA_DEPENDENCY_VERSION}
               QUIET
               COMPONENTS ${HACCABANA_DEPENDENCY_COMPONENTS})

  string(TOUPPER "${HACCABANA_DEPENDENCY_PACKAGE}" _DEP_UPPER)

  option(HACCabana_REQUIRE_${_DEP_UPPER}
         "Require ${HACCABANA_DEPENDENCY_PACKAGE} support"
         ${${HACCABANA_DEPENDENCY_PACKAGE}_FOUND})

  if(HACCabana_REQUIRE_${_DEP_UPPER})
    find_package(${HACCABANA_DEPENDENCY_PACKAGE}
                 ${HACCABANA_DEPENDENCY_VERSION}
                 REQUIRED
                 COMPONENTS ${HACCABANA_DEPENDENCY_COMPONENTS})
  endif()

  set(HACCabana_ENABLE_${_DEP_UPPER}
      ${${HACCABANA_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# Find Cabana (components are names, not imported target names)
find_package(Cabana REQUIRED COMPONENTS Grid Core)

# Validate Cabana build configuration
if(NOT Cabana_ENABLE_MPI)
  message(FATAL_ERROR "Cabana must be compiled with MPI (Cabana_ENABLE_MPI=ON).")
endif()

if(NOT Cabana_ENABLE_GRID)
  message(FATAL_ERROR "Cabana must be compiled with Grid support (Cabana_ENABLE_GRID=ON).")
endif()

# Optional dependency: Canopy
HACCabana_add_dependency(PACKAGE Canopy)
set_package_properties(Canopy PROPERTIES
  TYPE OPTIONAL
  PURPOSE "Fast multipole solver for far-field forces"
)

# Print git revision hash
find_package(Git QUIET)
if(Git_FOUND AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND           "${GIT_EXECUTABLE}" log --pretty=format:%H -n 1
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE   HACCabana_GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(HACCabana_GIT_COMMIT_HASH "Not a git repository")
endif()
message(STATUS "HACCabana Revision = '${HACCabana_GIT_COMMIT_HASH}'")

# Source code
add_subdirectory(src)

# Drivers
add_subdirectory(drivers)

# Package configuration files (HACCabanaConfig.cmake + version file)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HACCabanaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HACCabanaConfig.cmakein"
  "${CMAKE_CURRENT_BINARY_DIR}/HACCabanaConfig.cmake"
  @ONLY
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/HACCabanaConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/HACCabanaConfigVersion.cmake"
  DESTINATION "${HACCabana_INSTALL_PACKAGEDIR}"
)

feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)