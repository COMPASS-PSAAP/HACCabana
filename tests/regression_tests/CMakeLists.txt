if(NOT HACCabana_ENABLE_CANOPY)
  message(WARNING "Skipping regression tests because Canopy support is disabled.")
  return()
endif()

set(_regression_target HACCabana_Regression_ExactVsFMM)

add_executable(${_regression_target}
  exact_vs_fmm.cpp
)

target_link_libraries(${_regression_target}
  PRIVATE
    GTest::gtest
    HACCabana::haccabana
    Kokkos::kokkos
    MPI::MPI_CXX
)

target_compile_features(${_regression_target} PRIVATE cxx_std_17)

add_test(
  NAME regression_exact_vs_fmm_near_field
  COMMAND
    ${MPIEXEC_EXECUTABLE}
    ${MPIEXEC_NUMPROC_FLAG} 1
    $<TARGET_FILE:${_regression_target}>
)

set_tests_properties(regression_exact_vs_fmm_near_field
  PROPERTIES
    LABELS "regression"
)
