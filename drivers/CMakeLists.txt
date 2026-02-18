#------------------------------------------------------------------------------#
# Driver setup
#------------------------------------------------------------------------------#

# ---- helper to add one driver per backend -----------------------------------
function(haccabana_add_driver backend)
  set(target "HACCabana_Driver_${backend}")
  add_executable(${target} main.cxx)

  target_link_libraries(${target} PRIVATE HACCabana::haccabana Kokkos::kokkos)
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_definitions(${target} PRIVATE "HACCABANA_DRIVER_BACKEND_${backend}=1")

  # Output names
  # set_target_properties(${target} PROPERTIES OUTPUT_NAME "haccabana_driver_${backend}")

  # Add a "make driver_${backend}" convenience target
  add_custom_target("driver_${backend}" DEPENDS ${target})
endfunction()

# ---- create drivers for enabled backends ------------------------------------
# Always add serial
haccabana_add_driver(SERIAL)

if(Kokkos_ENABLE_OPENMP)
  haccabana_add_driver(OPENMP)
endif()

if(Kokkos_ENABLE_THREADS)
  haccabana_add_driver(THREADS)
endif()

if(Kokkos_ENABLE_CUDA)
  haccabana_add_driver(CUDA)
endif()

if(Kokkos_ENABLE_HIP)
  haccabana_add_driver(HIP)
endif()
