#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <iostream>

#include <Solver.h>

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

  // Kokkos::ScopeGuard initializes Kokkos and guarantees it is finalized.
  Kokkos::ScopeGuard scope_guard(argc, argv);

  // Program options:
  // Intput is either synthetic or from a file (optionally the answer is verified)

  int input_flag = 0;           // input file
  int verification_flag = 0;    // verification file (optional)
  int synthetic_data_flag = 0;  // generate synthetic data
  int timestep_flag = 0;        // timstep to advance to (required)
  int config_flag = 0;          // configuration file (optional)
  std::string input_filename = "";
  std::string verification_filename = "";
  char* t_value = NULL;
  std::string configuration_filename = "";
  int c;
  opterr = 0;

  while ((c = getopt (argc, argv, "i:v:st:c:x:")) != -1)
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
      case 's':
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
      case '?':
        if (optopt == 'i' || optopt == 'v' || optopt == 't' || optopt == 'c' )
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
    cout << "Incompatable options!" << endl;
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

  auto solver = HACCabana::createSolver<MemorySpace, ExecutionSpace>(step0);
  solver->setup(config_flag, configuration_filename);
  solver->advance();
  solver->setupParticles(input_flag, input_filename);
  solver->subCycle();

  // simulation and cosmology params
  // loads default
  // HACCabana::Parameters Params;

  // if (config_flag)
  // {
  //   Params.load_from_file(configuration_filename);
  // }

  // TimeStepper ts(
  //    Params.alpha,
	// 	 Params.a_in,
	// 	 Params.a_fin,
	// 	 Params.nsteps,
	// 	 Params.omega_matter,
	// 	 Params.omega_cdm,
	// 	 Params.omega_baryon,
	// 	 Params.omega_cb,
	// 	 Params.omega_nu,
	// 	 Params.omega_radiation,
	// 	 Params.f_nu_massless,
	// 	 Params.f_nu_massive,
	// 	 Params.w_de,
  //    Params.wa_de);
  
  // cout << "Advancing timestepper to step " << step0 << endl;
  // // get timestepper up to speed
  // for (int step = 0; step < step0; step++)
  //   ts.advanceFullStep();

  // // we're starting to subcycle after a PM kick
  // ts.advanceHalfStep();

  // // Setup particle data 
  // HACCabana::Particles P;

  // const float min_alive_pos = Params.oL;
  // const float max_alive_pos = Params.rL+Params.oL;

  // if (input_flag)
  // {
  //   cout << "Reading file: " << input_filename << endl;
  //   P.readRawData(input_filename);
  // }
  // else if (synthetic_data_flag)
  // {
  //   cout << "Generating synthetic data in range [" << min_alive_pos << "," << max_alive_pos << "] " 
  //        << "rL=" << Params.rL << " oL=" << Params.oL << endl;
  //   P.generateData(Params.np, Params.rL, Params.oL, MEAN_VEL);
  //   P.convert_phys2grid(Params.ng, Params.rL, ts.aa());
  // }

  // P.reorder(min_alive_pos, max_alive_pos); // TODO:assumes local extent equals the global extent
  // cout << "\t" << P.end-P.begin << " particles in [" << min_alive_pos << "," << max_alive_pos << "]" << endl;

  // HACCabana::ParticleActions PA(&P);
  // PA.subCycle(ts, Params.nsub, Params.gpscal, Params.rmax*Params.rmax, Params.rsm*Params.rsm, Params.cm_size, Params.oL, Params.rL+Params.oL);
  // don't check particles in boundary cells
  auto parameters = solver->parameters();
  const float dx_boundary = parameters.cm_size;
  const float min_alive_pos = parameters.oL;
  const float max_alive_pos = parameters.rL+parameters.oL;

  cout << "\tExcluding boundary cells of Linked Cell List.\n\tPrinting all particles within [" << parameters.oL+dx_boundary << "," << parameters.rL+parameters.oL-dx_boundary << ")" << endl;

  int num_p = solver->num_p();
  using Field = typename HACCabana::Solver<MemorySpace, ExecutionSpace>::particles_type::Field;
  cout << "\nPrinting result for " << num_p << " particles."  << endl;
  auto particles_h = solver->particles();
  auto particle_id = Cabana::slice<Field::ParticleID>( particles_h, "particle_id" );
  auto sort_data = Cabana::sortByKey( particle_id );
  Cabana::permute( sort_data, particles_h );
  auto position = Cabana::slice<Field::Position>( particles_h, "position" );
  for (int i=0; i<num_p; ++i)
  {
    if (position(i,0) >= min_alive_pos+dx_boundary &&\
        position(i,1) >= min_alive_pos+dx_boundary &&\
        position(i,2) >= min_alive_pos+dx_boundary &&\
        position(i,0) <  max_alive_pos-dx_boundary &&\
        position(i,1) <  max_alive_pos-dx_boundary &&\
        position(i,2) <  max_alive_pos-dx_boundary)
    {
      printf("P%d: pos %.2lf, %.2lf, %.2lf\n", i, position(i, 0), position(i, 1), position(i, 2));
    }
  }
  // verify against the answer from the simulation
  // --------------------------------------------------------------------------------------------------------------------------

  if (verification_flag)
  {
    using Field = typename HACCabana::Solver<MemorySpace, ExecutionSpace>::particles_type::Field;
    cout << "\nVerifying result." << endl;
    auto particles_h = solver->particles();
    auto particle_id = Cabana::slice<Field::ParticleID>( particles_h, "particle_id" );
    auto sort_data = Cabana::sortByKey( particle_id );

    // Create solver and load verification particles
    auto solver_ans = HACCabana::createSolver<MemorySpace, ExecutionSpace>(step0);
    solver_ans->setupParticles(1, verification_filename);
    // HACCabana::Particles<Kokkos::HostSpace, Kokkos::DefaultHostExecutionSpace> P_ans;
    // cout << "Reading file: " << verification_filename << endl;
    // P_ans.readRawData(verification_filename);

    Cabana::permute( sort_data, particles_h );
    auto particles_ans_h = solver_ans->particles();
    auto particle_id_ans = Cabana::slice<Field::ParticleID>( particles_ans_h, "particle_id_ans" );
    auto sort_data_ans = Cabana::sortByKey( particle_id_ans );
    Cabana::permute( sort_data_ans, particles_ans_h );

    auto position = Cabana::slice<Field::Position>( particles_h, "position" );
    auto position_ans = Cabana::slice<Field::Position>( particles_ans_h, "position_ans" );

    int num_p = solver->num_p();
    int num_p_ans = solver_ans->num_p();
    cout << "Checking " << num_p << " particles against " << num_p_ans << " answer particles." << endl;
    assert(num_p == num_p_ans);

    // don't check particles in boundary cells
    auto parameters = solver->parameters();
    const float dx_boundary = parameters.cm_size;
    const float min_alive_pos = parameters.oL;
    const float max_alive_pos = parameters.rL+parameters.oL;

    cout << "\tExcluding boundary cells of Linked Cell List.\n\tChecking all particles within [" << parameters.oL+dx_boundary << "," << parameters.rL+parameters.oL-dx_boundary << ")" << endl;

    int count = 0;
    int err_n = 0;
    for (int i=0; i<num_p_ans; ++i)
    {
      assert(particle_id(i) == particle_id_ans(i));
      bool is_inside = false;
      if (position(i,0) >= min_alive_pos+dx_boundary &&\
          position(i,1) >= min_alive_pos+dx_boundary &&\
          position(i,2) >= min_alive_pos+dx_boundary &&\
          position(i,0) <  max_alive_pos-dx_boundary &&\
          position(i,1) <  max_alive_pos-dx_boundary &&\
          position(i,2) <  max_alive_pos-dx_boundary)
      {
        is_inside = true;
        ++count;
      }
      if (is_inside && (!floatCompare(position(i,0),position_ans(i,0)) ||\
                        !floatCompare(position(i,1),position_ans(i,1)) ||\
                        !floatCompare(position(i,2),position_ans(i,2))))
      {
        ++err_n;
      }
    }
    cout << "\t" << err_n << " particles (out of " << count << ") have position relative error greater than " << MAX_ERR << endl;
  }

  return 0;
}
