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
// Solver class
//--------------------------------------
template <class MemorySpace, class ExecutionSpace>
class Solver
{
  public:
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;

    using parameters_type = Parameters;
    using particles_type = Particles<memory_space, execution_space>;
    using actions_type  = ParticleActions<particles_type>;
    using timestepper_type = TimeStepper;
    using aosoa_host_type = typename particles_type::aosoa_host_type; 

    Solver( const int step0 );

    void setup(const int config_flag, const std::string& configuration_filename);

    /**
     * Get timestepper up to speed and 
     * start to subcycle after a PM kick
     */
    void advance();

    /**
     * Initialize particle data,
     */
    void setupParticles(const int input_flag, const std::string& input_filename);

    void subCycle();

    int num_p() {return _particles.num_p;}

    // void solve( double t_final, int write_freq );
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

    Parameters parameters() {return _parameters;}
    aosoa_host_type particles() {return _particles.aosoa_host;}

  private:
    const int _step0;
    parameters_type _parameters;
    particles_type _particles;
    actions_type _actions;
    std::unique_ptr<timestepper_type> _timestepper;
};

template<class MemorySpace, class ExecutionSpace>
std::shared_ptr<Solver<MemorySpace, ExecutionSpace>>
createSolver( const int step0 )
{
    return std::make_shared<Solver<MemorySpace, ExecutionSpace>>(
            step0 );
}
   

} // end namespace HACCabana

// Include implementations
#include "Solver_impl.h"

#endif // SOLVER_H
