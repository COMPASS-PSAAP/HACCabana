#ifndef CANOPY_FORCE_SOLVER_H
#define CANOPY_FORCE_SOLVER_H

#include <Canopy_Core.hpp>

#include <mpi.h>

template <class AoSoAType, class Field>
class CanopyForceSolver
{
  private:
  // Variables needed for Canopy
  using memory_space = typename AoSoAType::memory_space;
  using execution_space = typename AoSoAType::execution_space;
  using MD = Canopy::ParticleMetadata<AoSoAType, Field::Position, Field::Gravity, Field::Potential, Field::Force>; 
  static constexpr num_coefficients = 6;
  std::shared_ptr<Canopy::Solver<execution_space, memory_space, MD,
                                 2, num_coefficients>> _solver;
  
  size_t _begin, _end;
  float _c;
  float _rmax2;
  float _rsm2;
  
  public:
  CanopyForceSolver() {}
  ~CanopyForceSolver() {}

  void setup_subcycle(AoSoAType& aosoa_device,
                      const size_t begin, const size_t end,
                      const float c, const float cm_size,
                      const float min_pos, const float max_pos,
                      const float rmax2, const float rsm2)
  {
    _begin = begin;
    _end = _end;
    _c = c;
    _rmax2 = rmax2;
    _rsm2 = rsm2;
    
    float dx = cm_size;
    float x_min = min_pos;
    float x_max = max_pos;

    float grid_delta[3] = {dx, dx, dx};
    float grid_min[3] = {x_min, x_min, x_min};
    float grid_max[3] = {x_max, x_max, x_max};

    // FMM parameters
    const int leaf_tiles = 64;
    const int reduction_factor = 2;

    _solver = Canopy::createSolver<execution_space, memory_space, MD, 2, num_coefficients>
        (grid_min, grid_max, leaf_tiles, reduction_factor, MPI_COMM_WORLD);
  }

  void updateVel(AoSoAType& aosoa_device)
  {
    printf("Solving with Canopy\n");
    // Solve for force
    _solver->solve(aosoa_device, 1);

    // Update velocity
    auto velocity = Cabana::slice<Field::Velocity>(aosoa_device, "velocity");
    auto force = Cabana::slice<Field::Force>(aosoa_device, "force");

    
    auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
    
    auto c = _c;
    auto vector_kick = KOKKOS_LAMBDA(const int s, const int a)
    {
        velocity.access(s,a,0) += force.access(s,a,0) * c;
        velocity.access(s,a,1) += force.access(s,a,0) * c;
        velocity.access(s,a,2) += force.access(s,a,0) * c;
    };

    Cabana::SimdPolicy<VECTOR_LENGTH, execution_space> simd_policy(_begin, _end);
    Cabana::simd_parallel_for( simd_policy, vector_kick, "kick" ); 

    Kokkos::fence();
  }
};

#endif // CANOPY_FORCE_SOLVER_H
