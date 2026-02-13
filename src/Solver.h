#ifndef SOLVER_H
#define SOLVER_H

#include <memory>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "Parameters.h"
#include "Particles.h"
#include "ParticleActions.h"
#include "TimeStepper.h"

namespace HACCabana
{
//--------------------------------------
// Base interface (non-templated)
//--------------------------------------
class SolverBase
{
  public:
    virtual ~SolverBase() = default;

    virtual void setup() = 0;
    virtual void step()  = 0;
    virtual void solve( double t_final, int write_freq ) = 0;
};

//--------------------------------------
// Templated solver
//--------------------------------------
template <class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space    = MemorySpace;

    using parameters_type = Parameters;
    using particles_type = Particles<memory_space, execution_space>;
    using actions_type   = ParticleActions<particles_type>;
    using timestepper_type = TimeStepper;

    Solver( int config_flag );

    void setup() override
    {
        // Example: initialize particle data, etc.
        // _particles.generateData(...);
        // _particles.convert_phys2grid(...);
    }

    void step() override
    {
        // Example: call your particle actions here.
        // _actions.subCycle(...);
    }

    void solve( double t_final, int write_freq ) override
    {
        // Example loop skeleton (replace with your actual time stepper logic)
        double t = 0.0;
        int step_count = 0;

        while ( t < t_final )
        {
            step();
            ++step_count;

            // if (write_freq > 0 && step_count % write_freq == 0) writeOutput(...);

            // advance time (placeholder)
            t += 1.0;
        }
    }

    particles_type& particles() { return _particles; }
    actions_type& actions() { return _actions; }

  private:
    parameters_type _parameters;
    particles_type _particles;
    actions_type   _actions;
    timestepper_type _timestepper;
};

std::shared_ptr<SolverBase>
createSolver( const std::string& device )
{
    // NOTE: your earlier snippet had "seral" typo; use "serial".
    if ( device == "serial" )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        using ExecSpace = Kokkos::Serial;
        using MemSpace  = Kokkos::HostSpace;
        return std::make_shared<Solver<ExecSpace, MemSpace, ModelOrder>>(
            std::forward<CtorArgs>(ctor_args)... );
#else
        throw std::runtime_error( "Serial backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "threads" )
    {
#if defined( KOKKOS_ENABLE_THREADS )
        using ExecSpace = Kokkos::Threads;
        using MemSpace  = Kokkos::HostSpace;
        return std::make_shared<Solver<ExecSpace, MemSpace, ModelOrder>>(
            std::forward<CtorArgs>(ctor_args)... );
#else
        throw std::runtime_error( "Threads backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "openmp" )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        using ExecSpace = Kokkos::OpenMP;
        using MemSpace  = Kokkos::HostSpace;
        return std::make_shared<Solver<ExecSpace, MemSpace, ModelOrder>>(
            std::forward<CtorArgs>(ctor_args)... );
#else
        throw std::runtime_error( "OpenMP backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "cuda" )
    {
#if defined( KOKKOS_ENABLE_CUDA )
        using ExecSpace = Kokkos::Cuda;
        using MemSpace  = Kokkos::CudaSpace; // or Kokkos::CudaUVMSpace
        return std::make_shared<Solver<ExecSpace, MemSpace, ModelOrder>>(
            std::forward<CtorArgs>(ctor_args)... );
#else
        throw std::runtime_error( "CUDA backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "hip" )
    {
#if defined( KOKKOS_ENABLE_HIP )
        using ExecSpace = Kokkos::HIP;
        using MemSpace  = Kokkos::HIPSpace; // or Kokkos::HIPManagedSpace
        return std::make_shared<Solver<ExecSpace, MemSpace, ModelOrder>>(
            std::forward<CtorArgs>(ctor_args)... );
#else
        throw std::runtime_error( "HIP backend not enabled in Kokkos" );
#endif
    }

    throw std::runtime_error( "Unknown device/backend string: " + device );
}

} // end namespace HACCabana

#endif // SOLVER_H
