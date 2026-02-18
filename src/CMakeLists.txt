# Generate config header into the build directory
configure_file(HACCabana_Config.hpp
               ${CMAKE_CURRENT_BINARY_DIR}/HACCabana_Config.hpp
               @ONLY)

set(HACCABANA_HEADERS
  Definitions.h
  Parameters.h
  ParticleActions.h
  ParticleActions_impl.h
  Particles.h
  Particle_impl.h
  TimeStepper.h
  Solver.h
  Solver_impl.h
)

set(HACCABANA_SOURCES
  Parameters.cxx
  TimeStepper.cxx
)

add_library(HACCabana ${HACCABANA_SOURCES})

target_include_directories(HACCabana
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>  # for generated config header if needed
    $<INSTALL_INTERFACE:include>
)

add_library(HACCabana::haccabana ALIAS HACCabana)
target_link_libraries(HACCabana PUBLIC Cabana::Core)

# Require C++17 for consumers and for building this target
target_compile_features(HACCabana PUBLIC cxx_std_17)

# Install the library and export its target
install(TARGETS HACCabana
  EXPORT HACCabana_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # harmless on non-Windows
)

install(EXPORT HACCabana_Targets
  FILE HACCabana_Targets.cmake
  NAMESPACE HACCabana::
  DESTINATION ${HACCabana_INSTALL_PACKAGEDIR}
)

# Install headers (including the generated header)
install(FILES ${HACCABANA_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/HACCabana_Config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
