#ifndef SOLVER_H
#define SOLVER_H

#include <memory>
#include <stdexcept>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <Kokkos_Core.hpp>

#include "Definitions.h"
#include "Parameters.h"
#include "Particles.h"
#include "ParticleActions.h"
#include "TimeStepper.h"

namespace HACCabana
{
//--------------------------------------
// Base interface
//--------------------------------------
class SolverBase
{
  public:
    struct ParticleData
    {
    using member_types = Cabana::MemberTypes<int64_t, float[3], float[3], int>;

        struct field
        {
            enum : int { ParticleID = 0, Position = 1, Velocity = 2, BinIndex = 3 };
        };
    };

    using member_types = typename ParticleData::member_types;
    using aosoa_host_type = Cabana::AoSoA<member_types, Kokkos::HostSpace, VECTOR_LENGTH>;

    virtual ~SolverBase() = default;

    virtual void setup(const int config_flag, const std::string& configuration_file) = 0;
    virtual void advance() = 0;
    virtual void setupParticles(int config_flag, const std::string& input_filename)  = 0;
    virtual void subCycle() = 0;
    virtual int num_p() = 0;

    virtual Parameters parameters() = 0;
    virtual aosoa_host_type particles() = 0;
};

//--------------------------------------
// Templated solver
//--------------------------------------
template <class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using data_types = DataTypes;
    using typename SolverBase::aosoa_host_type; 

    using parameters_type = Parameters;
    using particles_type = Particles<memory_space, execution_space, data_types>;
    using actions_type  = ParticleActions<particles_type>;
    using timestepper_type = TimeStepper;

    Solver( const int step0 );

    void setup(const int config_flag, const std::string& configuration_filename) override;

    /**
     * Get timestepper up to speed and 
     * start to subcycle after a PM kick
     */
    void advance() override;

    /**
     * Initialize particle data,
     */
    void setupParticles(const int input_flag, const std::string& input_filename) override;

    void subCycle() override;

    int num_p() override {return _particles.num_p;}

    // void solve( double t_final, int write_freq ) override;
    // {
    //     // Example loop skeleton (replace with your actual time stepper logic)
    //     double t = 0.0;
    //     int step_count = 0;

    //     while ( t < t_final )
    //     {
    //         step();
    //         ++step_count;

    //         // if (write_freq > 0 && step_count % write_freq == 0) writeOutput(...);

    //         // advance time (placeholder)
    //         t += 1.0;
    //     }
    // }

    Parameters parameters() override {return _parameters;}
    aosoa_host_type particles() override {return _particles.aosoa_host;}

  private:
    const int _step0;
    parameters_type _parameters;
    particles_type _particles;
    actions_type _actions;
    std::unique_ptr<timestepper_type> _timestepper;
};

extern template class Solver<Kokkos::HostSpace, Kokkos::Serial, DataTypes>;
extern template class Solver<Kokkos::CudaSpace, Kokkos::Cuda, DataTypes>;

template<class DataTypes>
std::shared_ptr<SolverBase<DataTypes>>
createSolver( const std::string& device, const int step0 )
{
    if ( device == "serial" )
    {
#if defined( KOKKOS_ENABLE_SERIAL )
        using ExecutionSpace = Kokkos::Serial;
        using MemorySpace = Kokkos::HostSpace;
        return std::make_shared<Solver<MemorySpace, ExecutionSpace, DataTypes>>(
            step0 );
#else
        throw std::runtime_error( "Serial backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "threads" )
    {
#if defined( KOKKOS_ENABLE_THREADS )
        using ExecutionSpace = Kokkos::Threads;
        using MemorySpace = Kokkos::HostSpace;
        return std::make_shared<Solver<MemorySpace, ExecutionSpace, DataTypes>>(
            step0 );
#else
        throw std::runtime_error( "Threads backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "openmp" )
    {
#if defined( KOKKOS_ENABLE_OPENMP )
        using ExecutionSpace = Kokkos::OpenMP;
        using MemorySpace = Kokkos::HostSpace;
        return std::make_shared<Solver<MemorySpace, ExecutionSpace, DataTypes>>(
            step0 );
#else
        throw std::runtime_error( "OpenMP backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "cuda" )
    {
#if defined( KOKKOS_ENABLE_CUDA )
        using ExecutionSpace = Kokkos::Cuda;
        using MemorySpace = Kokkos::CudaSpace;
        return std::make_shared<Solver<MemorySpace, ExecutionSpace, DataTypes>>(
            step0 );
#else
        throw std::runtime_error( "CUDA backend not enabled in Kokkos" );
#endif
    }
    else if ( device == "hip" )
    {
#if defined( KOKKOS_ENABLE_HIP )
        using ExecutionSpace = Kokkos::HIP;
        using MemorySpace = Kokkos::HIPSpace;
        return std::make_shared<Solver<MemorySpace, ExecutionSpace, DataTypes>>(
            step0 );
#else
        throw std::runtime_error( "HIP backend not enabled in Kokkos" );
#endif
    }

    throw std::runtime_error( "Unknown device/backend string: " + device );
}

} // end namespace HACCabana

#endif // SOLVER_H
