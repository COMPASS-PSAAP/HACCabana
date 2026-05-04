#ifndef CANOPY_FORCE_SOLVER_H
#define CANOPY_FORCE_SOLVER_H

#include <Canopy_Core.hpp>

#include "HACCabana_Definitions.h"

#include <mpi.h>

template <class AoSoAType, class Field>
class CanopyForceSolver
{
  public:
  static constexpr int P_ORDER = 10;

  private:
  // Variables needed for Canopy
  using memory_space = typename AoSoAType::memory_space;
  using execution_space = typename AoSoAType::execution_space;
  using Solver_t =
        Solver<memory_space, execution_space, float, P_ORDER, /*NComps=*/1>;
  std::shared_ptr<Solver_t> _solver;
  
  float _c;
  float _rmax2;
  float _rsm2;
  int _step;

  // Setup flag
  bool _is_setup;
  
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
      printf("Starting fmm solve...\n");

    _c = c;
    _rmax2 = rmax2;
    _rsm2 = rsm2;
    _step = 0;
    
    float dx = cm_size;

    // Buffer the original domain slightly without shifting it.
    float x_min = min_pos - dx;
    float x_max = max_pos + dx;

    // Tolerance on domain bounds
    std::array<double, 3> bb_tol = {0.05, 0.05, 0.05};
    
    // Min and max positions must be buffered slightly to avoid cell centers that
    // are out of domain bounds.
    std::array<float, 3> grid_min = {x_min, x_min, x_min};
    std::array<float, 3> grid_max = {x_max, x_max, x_max};

    // FMM parameters
    int ncrit = 32;
    double ncrit_tol = 0.15
    int max_dept = 15;
    int replication_depth = 3;

    _solver = Canopy::createSolver<memory_space, execution_space, float, P_ORDER, 1>
        (MPI_COMM_WORLD, ncrit, max_depth,
        bb_tol,
        ncrit_tol, replication_depth);

    _is_setup = false;
  }

  void updateVel(std::shared_ptr<AoSoAType> aosoa_device)
  {
    // First time we get particles we must setup solver
    if (!_is_setup)
    {
      solver->setup<Position, Charge>( aosoa_device, aosoa_device.size() );
      _is_setup = true;
    }
    // Canopy interprets positive scalars with the opposite sign convention
    // from HACCabana's gravitational force. Normalize force and potential
    // here so the rest of HACCabana always sees attractive gravity.
    _solver->solve(aosoa_device, false);

    // FMM solve for current state
    solver.template solve<Field::Position, Field::Charge>( aosoa_device,
                                              /*compute_gradient=*/true );

    // Update local positions and velocities from gradient.
    // Symplectic Euler: v += dt*g;  r += dt * drift * v.
    // Run on device so we write directly into the AoSoA slices.
    const int n_local = solver.num_local_particles();
    auto positions = Cabana::slice<Field::Position>( aosoa_device );
    auto velocities = Cabana::slice<Field::Velocity>( aosoa_device );
    auto grad = solver.gradient();
    const double dt_local = dt;
    const double drift_local = drift_multiplier;
    Kokkos::parallel_for(
        "MultiSolve::integrate",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, n_local ),
        KOKKOS_LAMBDA( int i ) {
            const double gx = grad( i, 0, 0 );
            const double gy = grad( i, 0, 1 );
            const double gz = grad( i, 0, 2 );
            velocities( i, 0 ) += dt_local * gx;
            velocities( i, 1 ) += dt_local * gy;
            velocities( i, 2 ) += dt_local * gz;
            positions( i, 0 ) +=
                dt_local * drift_local * velocities( i, 0 );
            positions( i, 1 ) +=
                dt_local * drift_local * velocities( i, 1 );
            positions( i, 2 ) +=
                dt_local * drift_local * velocities( i, 2 );
        } );
    Kokkos::fence();

    auto grad = Cabana::slice<Field::Force>( *aosoa_device, "force" );

    Kokkos::parallel_for(
        "convert_canopy_gravity_convention",
        Kokkos::RangePolicy<execution_space>( 0, aosoa_device->size() ),
        KOKKOS_LAMBDA( const int i )
        {
          force( i, 0 ) = -force( i, 0 );
          force( i, 1 ) = -force( i, 1 );
          force( i, 2 ) = -force( i, 2 );
          potential( i ) = -potential( i );
        } );

    Kokkos::fence();

    // Update velocity
    auto velocity = Cabana::slice<Field::Velocity>( *aosoa_device, "velocity" );

    auto c = _c;
    auto vector_kick = KOKKOS_LAMBDA(const int s, const int a)
    {
        velocity.access(s,a,0) += force.access(s,a,0) * c;
        velocity.access(s,a,1) += force.access(s,a,1) * c;
        velocity.access(s,a,2) += force.access(s,a,2) * c;
    };

    Cabana::SimdPolicy<VECTOR_LENGTH, execution_space> simd_policy(0, aosoa_device->size());
    Cabana::simd_parallel_for( simd_policy, vector_kick, "kick" );

    Kokkos::fence();

    _step++;
  }
};

#endif // CANOPY_FORCE_SOLVER_H
