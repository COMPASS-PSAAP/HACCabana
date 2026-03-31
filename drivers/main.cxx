#include <ctype.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <HACCabana_Solver.h>

#include <mpi.h>

#if defined(HACCABANA_DRIVER_BACKEND_CUDA)
  using ExecutionSpace = Kokkos::Cuda;
  using MemorySpace = Kokkos::CudaSpace;
#elif defined(HACCABANA_DRIVER_BACKEND_HIP)
  using ExecutionSpace = Kokkos::HIP;
  using MemorySpace = Kokkos::HIPSpace;
#elif defined(HACCABANA_DRIVER_BACKEND_OPENMP)
  using ExecutionSpace = Kokkos::OpenMP;
  using MemorySpace = Kokkos::HostSpace;
#elif defined(HACCABANA_DRIVER_BACKEND_THREADS)
  using ExecutionSpace = Kokkos::Threads;
  using MemorySpace = Kokkos::HostSpace;
#else
  using ExecutionSpace = Kokkos::Serial;
  using MemorySpace = Kokkos::HostSpace;
#endif

using namespace std;

inline bool floatCompare(float f1, float f2) {
  static constexpr auto epsilon = MAX_ERR;
  if (fabs(f1 - f2) <= epsilon)
    return true;
  return fabs(f1 - f2) <= epsilon * fmax(fabs(f1), fabs(f2));
}

int main( int argc, char* argv[] )
{
  MPI_Init( &argc, &argv );
  int return_code = 0;
  try
  {
  // Kokkos::ScopeGuard initializes Kokkos and guarantees it is finalized.
  Kokkos::ScopeGuard scope_guard(argc, argv);

  // Program options:
  // Intput is either synthetic or from a file (optionally the answer is verified)

  int input_flag = 0;           // input file
  int verification_flag = 0;    // verification file (optional)
  int synthetic_data_flag = 0;  // generate synthetic data
  int timestep_flag = 0;        // timstep to advance to (required)
  int config_flag = 0;          // configuration file (optional)
  int print_positions_flag = 0; // print final particle positions (optional)
  int leaf_tiles_flag = 0;      // Canopy leaf tiles (FMM only)
  int reduction_factor_flag = 0; // Canopy reduction factor (FMM only)
  std::size_t num_particles = 0;
  int num_substeps = 0;
  int leaf_tiles = 16;
  int reduction_factor = 2;
  HACCabana::force_solver_type force_solver = HACCabana::force_solver_type::p3m;
  std::string input_filename = "";
  std::string verification_filename = "";
  std::string force_solver_name = "p3m";
  char* t_value = NULL;
  std::string configuration_filename = "";
  int c;
  opterr = 0;
  static struct option long_options[] = {
    {"print-positions", no_argument, nullptr, 'P'},
    {nullptr, 0, nullptr, 0}
  };
  int option_index = 0;

  while ((c = getopt_long(argc, argv, "i:v:dt:c:p:s:f:Pl:r:", long_options,
                          &option_index)) != -1)
    switch (c)
    {
      case 'i':
        input_flag = 1;
        input_filename = optarg;
        break;
      case 'v':
        verification_flag = 1;
        verification_filename = optarg;
        break;
      case 'd':
        synthetic_data_flag = 1;
        break;
      case 't':
        timestep_flag = 1;
        t_value = optarg;
        break;
      case 'c':
        config_flag = 1;
        configuration_filename = optarg;
        break;
      case 'p':
        num_particles = std::stoi(optarg);
        break;
      case 's':
        num_substeps = std::stoi(optarg);
        break;
      case 'f':
        force_solver_name = optarg;
        force_solver = HACCabana::parse_force_solver_type(force_solver_name);
        break;
      case 'P':
        print_positions_flag = 1;
        break;
      case 'l':
        leaf_tiles_flag = 1;
        leaf_tiles = std::stoi(optarg);
        break;
      case 'r':
        reduction_factor_flag = 1;
        reduction_factor = std::stoi(optarg);
        break;
      case '?':
        if (optopt == 'i' || optopt == 'v' || optopt == 't' || optopt == 'c' ||
            optopt == 'p' || optopt == 's' || optopt == 'f' || optopt == 'l' ||
            optopt == 'r')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
      default:
        abort();
    }
  if (!(input_flag || synthetic_data_flag))
  {
    cout << "No input or synthetic flag." << endl;
    return 1;
  }
  if (input_flag && synthetic_data_flag || verification_flag && synthetic_data_flag) 
  {
    cout << "Incompatible options!" << endl;
    return 1;
  }

  int step0;  
  if (!timestep_flag)
  {
    cout << "Timestep required!" << endl;
    return 1;
  }
  else
  {
    step0 = atoi(t_value);
  }

  if (leaf_tiles <= 0)
    throw std::runtime_error("Option '-l' requires a positive integer.");

  if (reduction_factor <= 0)
    throw std::runtime_error("Option '-r' requires a positive integer.");

  if ((leaf_tiles_flag || reduction_factor_flag) &&
      force_solver != HACCabana::force_solver_type::fmm)
    throw std::runtime_error(
        "Options '-l' and '-r' require the FMM solver. Use '-f fmm'." );

#ifndef HACCabana_ENABLE_CANOPY
  if (force_solver == HACCabana::force_solver_type::fmm)
    throw std::runtime_error(
        "Requested '-f fmm', but HACCabana_ENABLE_CANOPY is not defined." );
#endif

  auto solver = HACCabana::createSolver<MemorySpace, ExecutionSpace>(step0, force_solver);
  if (leaf_tiles_flag)
    solver->setLeafTiles(leaf_tiles);
  if (reduction_factor_flag)
    solver->setReductionFactor(reduction_factor);
  solver->setup(config_flag, configuration_filename, num_particles, num_substeps);
  solver->advance();
  solver->setupParticles(input_flag, input_filename);
  solver->subCycle();

  if (print_positions_flag)
  {
    // don't check particles in boundary cells
    auto parameters = solver->parameters();
    const float dx_boundary = parameters.cm_size;
    const float min_alive_pos = parameters.oL;
    const float max_alive_pos = parameters.rL+parameters.oL;

    // cout << "\tExcluding boundary cells of Linked Cell List.\n\tPrinting all particles within [" << parameters.oL+dx_boundary << "," << parameters.rL+parameters.oL-dx_boundary << ")" << endl;

    int num_p = solver->num_p();
    using Field = typename HACCabana::Solver<MemorySpace, ExecutionSpace>::particles_type::Field;
    cout << "\nPrinting final particle positions:"  << endl;
    auto particles_h = solver->data();
    auto particle_id = Cabana::slice<Field::ParticleID>( particles_h, "particle_id" );
    auto position = Cabana::slice<Field::Position>( particles_h, "position" );

    std::vector<int> sorted_indices( num_p );
    std::iota( sorted_indices.begin(), sorted_indices.end(), 0 );
    std::sort( sorted_indices.begin(), sorted_indices.end(),
               [&]( const int lhs, const int rhs )
               { return particle_id( lhs ) < particle_id( rhs ); } );

    for ( const int i : sorted_indices )
    {
      if (position(i,0) >= min_alive_pos+dx_boundary &&\
          position(i,1) >= min_alive_pos+dx_boundary &&\
          position(i,2) >= min_alive_pos+dx_boundary &&\
          position(i,0) <  max_alive_pos-dx_boundary &&\
          position(i,1) <  max_alive_pos-dx_boundary &&\
          position(i,2) <  max_alive_pos-dx_boundary)
      {
        printf("p(%.4lf, %.4lf, %.4lf)\n", position(i, 0), position(i, 1), position(i, 2));
      }
    }
  }

  } // Kokkos scopeguard
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return_code = 1;
  }
  MPI_Finalize();

  return return_code;
}
