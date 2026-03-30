#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <mpi.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include "HACCabana_CanopyForceSolver.h"
#include "HACCabana_ExactForceSolver.h"
#include "HACCabana_Particles.h"

namespace
{

using execution_space = Kokkos::DefaultHostExecutionSpace;
using memory_space = typename execution_space::memory_space;
using particles_type = HACCabana::Particles<memory_space, execution_space>;
using aosoa_type = typename particles_type::aosoa_type;
using aosoa_host_type = typename particles_type::aosoa_host_type;
using Field = typename particles_type::Field;

constexpr float min_pos = 0.0f;
constexpr float max_pos = 1.0f;
constexpr float cm_size = 0.1f;
constexpr float force_tolerance = 1.0e-2f;

aosoa_host_type createDomainParticles( const std::size_t num_particles )
{
  aosoa_host_type particles( "regression_particles", num_particles );

  auto particle_id = Cabana::slice<Field::ParticleID>( particles, "particle_id" );
  auto position = Cabana::slice<Field::Position>( particles, "position" );
  auto velocity = Cabana::slice<Field::Velocity>( particles, "velocity" );
  auto force = Cabana::slice<Field::Force>( particles, "force" );
  auto gravity = Cabana::slice<Field::Gravity>( particles, "gravity" );
  auto potential = Cabana::slice<Field::Potential>( particles, "potential" );
  auto bin_index = Cabana::slice<Field::BinIndex>( particles, "bin_index" );

  const float domain_span = max_pos - min_pos;
  const int particles_per_dim = std::max<int>(
      1, static_cast<int>( std::ceil( std::cbrt( num_particles ) ) ) );
  const float padding = 0.1f * domain_span;
  const float active_min = min_pos + padding;
  const float active_max = max_pos - padding;
  const float active_span = active_max - active_min;
  const float spacing =
      ( particles_per_dim > 1 )
          ? active_span / static_cast<float>( particles_per_dim - 1 )
          : 0.0f;

  for ( std::size_t i = 0; i < num_particles; ++i )
  {
    const int ix = i % particles_per_dim;
    const int iy = ( i / particles_per_dim ) % particles_per_dim;
    const int iz = i / ( particles_per_dim * particles_per_dim );

    particle_id( i ) = i;
    gravity( i ) = 1.0f + 0.125f * static_cast<float>( i % 7 );
    potential( i ) = 0.0f;
    bin_index( i ) = 0;

    position( i, 0 ) = active_min + ix * spacing;
    position( i, 1 ) = active_min + iy * spacing;
    position( i, 2 ) = active_min + iz * spacing;

    velocity( i, 0 ) = 0.0f;
    velocity( i, 1 ) = 0.0f;
    velocity( i, 2 ) = 0.0f;

    force( i, 0 ) = 0.0f;
    force( i, 1 ) = 0.0f;
    force( i, 2 ) = 0.0f;
  }

  return particles;
}

std::shared_ptr<aosoa_type> copyToDevice( const aosoa_host_type& particles_h,
                                          const std::string& label )
{
  auto particles_d =
      std::make_shared<aosoa_type>( label, particles_h.size() );
  Cabana::deep_copy( *particles_d, particles_h );
  return particles_d;
}

void expectParticlesSpanDomain( const aosoa_host_type& particles_h )
{
  auto position = Cabana::slice<Field::Position>( particles_h, "position" );

  const float grid_min = min_pos - cm_size;
  const float grid_max = max_pos + cm_size;
  const float cell_size =
      ( grid_max - grid_min ) /
      ( CanopyForceSolver<aosoa_type, Field>::leaf_tiles * 2.0f );

  for ( int d = 0; d < 3; ++d )
  {
    float min_coord = position( 0, d );
    float max_coord = position( 0, d );
    for ( std::size_t i = 1; i < particles_h.size(); ++i )
    {
      min_coord = std::min( min_coord, position( i, d ) );
      max_coord = std::max( max_coord, position( i, d ) );
    }

    ASSERT_GE( min_coord, min_pos );
    ASSERT_LT( max_coord, max_pos );
    ASSERT_GT( max_coord - min_coord, 4.0f * cell_size )
        << "Particles do not span enough of the domain in dimension " << d;
  }
}

} // end anonymous namespace

TEST( RegressionExactVsFMM, NearFieldForcesMatch )
{
  const std::size_t num_particles = 50;

  int mpi_size = 0;
  MPI_Comm_size( MPI_COMM_WORLD, &mpi_size );
  ASSERT_EQ( mpi_size, 1 );
  ASSERT_GT( num_particles, 1u );

  auto particles_h = createDomainParticles( num_particles );
  ASSERT_EQ( particles_h.size(), num_particles );
  expectParticlesSpanDomain( particles_h );

  auto exact_particles = copyToDevice( particles_h, "exact_particles" );
  auto fmm_particles = copyToDevice( particles_h, "fmm_particles" );

  ExactForceSolver<aosoa_type, Field> exact_solver;
  CanopyForceSolver<aosoa_type, Field> fmm_solver;

  constexpr float c = 1.0f;
  constexpr float rmax2 = 0.0f;
  constexpr float rsm2 = 0.0f;

  exact_solver.setup_subcycle( *exact_particles, c, cm_size, min_pos, max_pos,
                               rmax2, rsm2 );
  fmm_solver.setup_subcycle( *fmm_particles, c, cm_size, min_pos, max_pos,
                             rmax2, rsm2 );

  exact_solver.updateVel( exact_particles );
  fmm_solver.updateVel( fmm_particles );
  Kokkos::fence();

  auto exact_h =
      Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), *exact_particles );
  auto fmm_h =
      Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), *fmm_particles );

  auto exact_force = Cabana::slice<Field::Force>( exact_h, "exact_force" );
  auto fmm_force = Cabana::slice<Field::Force>( fmm_h, "fmm_force" );
  auto exact_id = Cabana::slice<Field::ParticleID>( exact_h, "exact_id" );
  auto fmm_id = Cabana::slice<Field::ParticleID>( fmm_h, "fmm_id" );

  std::vector<std::size_t> exact_order( particles_h.size() );
  std::vector<std::size_t> fmm_order( particles_h.size() );
  std::iota( exact_order.begin(), exact_order.end(), 0 );
  std::iota( fmm_order.begin(), fmm_order.end(), 0 );

  auto by_exact_id = [&]( const std::size_t lhs, const std::size_t rhs )
  { return exact_id( lhs ) < exact_id( rhs ); };
  auto by_fmm_id = [&]( const std::size_t lhs, const std::size_t rhs )
  { return fmm_id( lhs ) < fmm_id( rhs ); };

  std::sort( exact_order.begin(), exact_order.end(), by_exact_id );
  std::sort( fmm_order.begin(), fmm_order.end(), by_fmm_id );

  for ( std::size_t i = 0; i < particles_h.size(); ++i )
  {
    ASSERT_EQ( exact_id( exact_order[i] ), fmm_id( fmm_order[i] ) )
        << "Particle ordering differs between exact and FMM results.";

    for ( int d = 0; d < 3; ++d )
    {
      EXPECT_NEAR( exact_force( exact_order[i], d ),
                   fmm_force( fmm_order[i], d ), force_tolerance )
          << "Mismatch for particle id " << exact_id( exact_order[i] )
          << " component " << d;
    }
  }
}

TEST( Particles, ReorderDropsOutOfBoundsParticlesAndPreservesFields )
{
  particles_type particles;
  particles.aosoa_host = aosoa_host_type( "reorder_particles", 4 );

  auto particle_id =
      Cabana::slice<Field::ParticleID>( particles.aosoa_host, "particle_id" );
  auto position =
      Cabana::slice<Field::Position>( particles.aosoa_host, "position" );
  auto velocity =
      Cabana::slice<Field::Velocity>( particles.aosoa_host, "velocity" );
  auto force = Cabana::slice<Field::Force>( particles.aosoa_host, "force" );
  auto gravity =
      Cabana::slice<Field::Gravity>( particles.aosoa_host, "gravity" );
  auto potential =
      Cabana::slice<Field::Potential>( particles.aosoa_host, "potential" );
  auto bin_index =
      Cabana::slice<Field::BinIndex>( particles.aosoa_host, "bin_index" );

  const std::array<std::array<float, 3>, 4> positions = {{
      {{0.25f, 0.25f, 0.25f}},
      {{-0.10f, 0.40f, 0.40f}},
      {{0.75f, 0.75f, 0.75f}},
      {{1.10f, 0.60f, 0.60f}},
  }};

  for ( int i = 0; i < 4; ++i )
  {
    particle_id( i ) = 100 + i;
    gravity( i ) = 10.0f + i;
    potential( i ) = 20.0f + i;
    bin_index( i ) = 30 + i;

    for ( int d = 0; d < 3; ++d )
    {
      position( i, d ) = positions[i][d];
      velocity( i, d ) = 40.0f + 10.0f * i + d;
      force( i, d ) = 70.0f + 10.0f * i + d;
    }
  }

  particles.reorder( 0.0f, 1.0f );

  ASSERT_EQ( particles.aosoa_host.size(), 2u );

  auto kept_particle_id =
      Cabana::slice<Field::ParticleID>( particles.aosoa_host, "particle_id" );
  auto kept_position =
      Cabana::slice<Field::Position>( particles.aosoa_host, "position" );
  auto kept_velocity =
      Cabana::slice<Field::Velocity>( particles.aosoa_host, "velocity" );
  auto kept_force =
      Cabana::slice<Field::Force>( particles.aosoa_host, "force" );
  auto kept_gravity =
      Cabana::slice<Field::Gravity>( particles.aosoa_host, "gravity" );
  auto kept_potential =
      Cabana::slice<Field::Potential>( particles.aosoa_host, "potential" );
  auto kept_bin_index =
      Cabana::slice<Field::BinIndex>( particles.aosoa_host, "bin_index" );

  std::vector<int64_t> kept_ids;
  kept_ids.reserve( particles.aosoa_host.size() );
  for ( std::size_t i = 0; i < particles.aosoa_host.size(); ++i )
    kept_ids.push_back( kept_particle_id( i ) );
  std::sort( kept_ids.begin(), kept_ids.end() );

  ASSERT_EQ( kept_ids, ( std::vector<int64_t>{ 100, 102 } ) );

  for ( std::size_t i = 0; i < particles.aosoa_host.size(); ++i )
  {
    const int original_index = static_cast<int>( kept_particle_id( i ) - 100 );
    ASSERT_GE( original_index, 0 );
    ASSERT_LT( original_index, 4 );

    for ( int d = 0; d < 3; ++d )
    {
      EXPECT_FLOAT_EQ( kept_position( i, d ), positions[original_index][d] );
      EXPECT_FLOAT_EQ( kept_velocity( i, d ),
                       40.0f + 10.0f * original_index + d );
      EXPECT_FLOAT_EQ( kept_force( i, d ),
                       70.0f + 10.0f * original_index + d );
    }

    EXPECT_FLOAT_EQ( kept_gravity( i ), 10.0f + original_index );
    EXPECT_FLOAT_EQ( kept_potential( i ), 20.0f + original_index );
    EXPECT_EQ( kept_bin_index( i ), 30 + original_index );
  }
}

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  Kokkos::initialize( argc, argv );
  ::testing::InitGoogleTest( &argc, argv );

  const int test_result = RUN_ALL_TESTS();

  Kokkos::finalize();
  MPI_Finalize();

  return test_result;
}
