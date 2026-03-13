# Generate config header into the build directory
configure_file(HACCabana_Config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/HACCabana_Config.h)

set(HACCABANA_HEADERS
  Definitions.h
  ForceSolvers.h
  Parameters.h
  ParticleActions.h
  Particles.h
  Particle_impl.h
  TimeStepper.h
  Solver.h
)

if(HACCabana_ENABLE_CANOPY)
  list(APPEND HACCABANA_HEADERS
    CanopyForceSolver.h
    )
endif()

set(HACCABANA_SOURCES
  Parameters.cxx
  TimeStepper.cxx
)

add_library(HACCabana ${HACCABANA_SOURCES})

target_include_directories(HACCabana
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>  # for generated config header
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

add_library(HACCabana::haccabana ALIAS HACCabana)
target_link_libraries(HACCabana PUBLIC Cabana::Core)

if(HACCabana_ENABLE_CANOPY)
  target_link_libraries(HACCabana INTERFACE Canopy::Canopy)
endif()


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

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/HACCabana_Config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
