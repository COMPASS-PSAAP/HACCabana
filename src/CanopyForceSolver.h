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
  using MD = Canopy::ParticleMetadata<AoSoAType, float, Field::Position, Field::Gravity, Field::Potential, Field::Force>; 
  static constexpr int num_coefficients = 6;
  std::shared_ptr<Canopy::Solver<memory_space, execution_space, MD,
                                 2, num_coefficients>> _solver;
  
  float _c;
  float _rmax2;
  float _rsm2;
  int _step;
  
  public:
  CanopyForceSolver() {}
  ~CanopyForceSolver() {}

  void setup_subcycle(AoSoAType& aosoa_device,
                      const float c, const float cm_size,
                      const float min_pos, const float max_pos,
                      const float rmax2, const float rsm2)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
      printf("Solving with Canopy...\n");

    _c = c;
    _rmax2 = rmax2;
    _rsm2 = rsm2;
    _step = 0;
    
    float dx = cm_size;
    float x_min = min_pos - dx;
    float x_max = max_pos + dx;
    
    // Min and max positions must be buffered slightly to avoid cell centers that
    // are out of domain bounds.
    std::array<float, 3> grid_min = {x_min, x_min, x_min};
    std::array<float, 3> grid_max = {x_max, x_max, x_max};

    // FMM parameters
    const int leaf_tiles = 16;
    const int reduction_factor = 4;

    _solver = Canopy::createSolver<memory_space, execution_space, MD, 2, num_coefficients>
        (grid_min, grid_max, leaf_tiles, reduction_factor, MPI_COMM_WORLD);
  }

  void updateVel(std::shared_ptr<AoSoAType> aosoa_device)
  {
    // Solve for force
    _solver->solve(aosoa_device, 0);
        
    // Update velocity
    auto velocity = Cabana::slice<Field::Velocity>(*aosoa_device, "velocity");
    auto force = Cabana::slice<Field::Force>(*aosoa_device, "force");

    auto c = _c;
    auto vector_kick = KOKKOS_LAMBDA(const int s, const int a)
    {
        velocity.access(s,a,0) += force.access(s,a,0) * c;
        velocity.access(s,a,1) += force.access(s,a,0) * c;
        velocity.access(s,a,2) += force.access(s,a,0) * c;
    };

    Cabana::SimdPolicy<VECTOR_LENGTH, execution_space> simd_policy(0, aosoa_device->size());
    Cabana::simd_parallel_for( simd_policy, vector_kick, "kick" );

    Kokkos::fence();

    _step++;
  }
};

#endif // CANOPY_FORCE_SOLVER_H
