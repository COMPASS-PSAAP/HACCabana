#ifndef PARTICLES_H
#define PARTICLES_H

#include <string>

#include <Cabana_Core.hpp>

#include "Definitions.h"

namespace HACCabana
{

  template <class MemorySpace, class ExecutionSpace>
  class Particles
  {
    public:
      struct ParticleData
      {
      using member_types = Cabana::MemberTypes<int64_t, float[3], float[3], int>;

          struct Field
          {
              enum : int { ParticleID = 0, Position = 1, Velocity = 2, BinIndex = 3 };
          };
      };
      using memory_space = MemorySpace;
      using execution_space = ExecutionSpace;
      using member_types = typename ParticleData::member_types;
      using Field = typename ParticleData::Field;
      using aosoa_type = Cabana::AoSoA<member_types, memory_space, VECTOR_LENGTH>;
      using aosoa_host_type = Cabana::AoSoA<member_types, Kokkos::HostSpace, VECTOR_LENGTH>;

      size_t num_p = 0;
      size_t begin = 0;
      size_t end = 0;
      aosoa_type aosoa;
      aosoa_host_type aosoa_host;

      Particles();
      ~Particles();
      void generateData(const int np, const float rl, const float ol, const float mean_vel);
      void convert_phys2grid(int ng, float rL, float a);
      void readRawData(std::string file_name);
      void reorder(const float min_pos, const float max_pos);
  };
}

// Include implementations
#include "Particles_impl.h"

#endif
