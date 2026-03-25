#ifndef EXACT_FORCE_SOLVER_HPP
#define EXACT_FORCE_SOLVER_HPP

#include <memory>

#include <Cabana_Core.hpp>

#include "HACCabana_Definitions.h"

template <class AoSoAType, class Field>
class ExactForceSolver
{
  private:
  using memory_space = typename AoSoAType::memory_space;
  using execution_space = typename AoSoAType::execution_space;
  
  float _c;

  public:
  ExactForceSolver() {}
  ~ExactForceSolver() {}

  

  void setup_subcycle( AoSoAType& aosoa_device, const float c,
                       const float cm_size, const float min_pos,
                       const float max_pos, const float rmax2,
                       const float rsm2 )
  {
    // (void)aosoa_device;
    _c = c;
    // (void)cm_size;
    // (void)min_pos;
    // (void)max_pos;
    // (void)rmax2;
    // (void)rsm2;
  }

  void updateVel( std::shared_ptr<AoSoAType> aosoa_device )
  {
    auto gravity = Cabana::slice<Field::Gravity>( *aosoa_device, "gravity" );
    auto force = Cabana::slice<Field::Force>( *aosoa_device, "force" );
    auto position = Cabana::slice<Field::Position>( *aosoa_device, "position" );
    const std::size_t num_particles = aosoa_device->size();

    Kokkos::parallel_for(
        "exact_force",
        Kokkos::RangePolicy<execution_space>( 0, num_particles ),
        KOKKOS_LAMBDA( const int a )
        {
          const float x0 = position( a, 0 );
          const float y0 = position( a, 1 );
          const float z0 = position( a, 2 );
          const float g0 = gravity( a );

          double fx = 0.0;
          double fy = 0.0;
          double fz = 0.0;

          for ( std::size_t b = 0; b < num_particles; ++b )
          {
            if ( b == static_cast<std::size_t>( a ) )
              continue;

            const double dx = static_cast<double>( position( b, 0 ) ) -
                              static_cast<double>( x0 );
            const double dy = static_cast<double>( position( b, 1 ) ) -
                              static_cast<double>( y0 );
            const double dz = static_cast<double>( position( b, 2 ) ) -
                              static_cast<double>( z0 );
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double dist_inv = 1.0 / Kokkos::sqrt( r2 );
            const double dist_inv3 = dist_inv * dist_inv * dist_inv;
            const double fp = static_cast<double>( g0 ) *
                              static_cast<double>( gravity( b ) ) * dist_inv3;

            fx += fp * dx;
            fy += fp * dy;
            fz += fp * dz;
          }

          force( a, 0 ) = static_cast<float>( fx );
          force( a, 1 ) = static_cast<float>( fy );
          force( a, 2 ) = static_cast<float>( fz );
        } );

    Kokkos::fence();
        
    // Update velocity
    auto velocity = Cabana::slice<Field::Velocity>(*aosoa_device, "velocity");
    

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
  }
};

#endif
