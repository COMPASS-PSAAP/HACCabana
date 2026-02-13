

include(CMakeFindDependencyMacro)

# Required deps for HACCabana:: targets
find_dependency(Kokkos REQUIRED)
find_dependency(Cabana REQUIRED COMPONENTS Grid Core)

# If HACCabana was built with canopy enabled, require it for consumers too
set(HACCabana_ENABLE_CANOPY 0)
if(HACCabana_ENABLE_CANOPY)
  find_dependency(canopy REQUIRED)
endif()

# Import the installed targets for this package
include("${CMAKE_CURRENT_LIST_DIR}/HACCabana_Targets.cmake")
