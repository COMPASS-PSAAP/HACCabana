#ifndef EXACT_FORCE_SOLVER_HPP
#define EXACT_FORCE_SOLVER_HPP

#include <memory>

#include <Cabana_Core.hpp>

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

    using team_policy = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename team_policy::member_type;

    Kokkos::parallel_for(
        "exact_force",
        team_policy( num_particles, Kokkos::AUTO() ),
        KOKKOS_LAMBDA( const member_type& team )
        {
          const int a = team.league_rank();
          const float x0 = position( a, 0 );
          const float y0 = position( a, 1 );
          const float z0 = position( a, 2 );
          const float g0 = gravity( a );

          float fx = 0.0f;
          float fy = 0.0f;
          float fz = 0.0f;

          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange( team, num_particles ),
              [&]( const int b, float& local_fx )
              {
                if ( b == a )
                  return;

                const float dx = position( b, 0 ) - x0;
                const float dy = position( b, 1 ) - y0;
                const float dz = position( b, 2 ) - z0;
                const float r2 = dx * dx + dy * dy + dz * dz;
                const float dist_inv = 1.0f / Kokkos::sqrt( r2 );
                const float dist_inv3 = dist_inv * dist_inv * dist_inv;
                local_fx += g0 * gravity( b ) * dist_inv3 * dx;
              },
              fx );

          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange( team, num_particles ),
              [&]( const int b, float& local_fy )
              {
                if ( b == a )
                  return;

                const float dx = position( b, 0 ) - x0;
                const float dy = position( b, 1 ) - y0;
                const float dz = position( b, 2 ) - z0;
                const float r2 = dx * dx + dy * dy + dz * dz;
                const float dist_inv = 1.0f / Kokkos::sqrt( r2 );
                const float dist_inv3 = dist_inv * dist_inv * dist_inv;
                local_fy += g0 * gravity( b ) * dist_inv3 * dy;
              },
              fy );

          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange( team, num_particles ),
              [&]( const int b, float& local_fz )
              {
                if ( b == a )
                  return;

                const float dx = position( b, 0 ) - x0;
                const float dy = position( b, 1 ) - y0;
                const float dz = position( b, 2 ) - z0;
                const float r2 = dx * dx + dy * dy + dz * dz;
                const float dist_inv = 1.0f / Kokkos::sqrt( r2 );
                const float dist_inv3 = dist_inv * dist_inv * dist_inv;
                local_fz += g0 * gravity( b ) * dist_inv3 * dz;
              },
              fz );

          Kokkos::single( Kokkos::PerTeam( team ), [&]()
          {
            force( a, 0 ) = fx;
            force( a, 1 ) = fy;
            force( a, 2 ) = fz;
          } );
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
