#ifndef SOLVER_H
#define SOLVER_H

#include <memory>
#include <stdexcept>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "HACCabana_Config.h"

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

    Solver( const int step0 )
        : _step0(step0)
        , _parameters()
        , _particles()
        , _actions( &_particles )
    {
        
    }

    void setup(const int config_flag, const std::string& configuration_filename, const std::size_t num_particles,
        const int num_substeps)
    {
        if (config_flag)
        {
            _parameters.load_from_file(configuration_filename, num_particles, num_substeps);
        }
        _timestepper = std::make_unique<timestepper_type>(
                _parameters.alpha,
                _parameters.a_in,
                _parameters.a_fin,
                _parameters.nsteps,
                _parameters.omega_matter,
                _parameters.omega_cdm,
                _parameters.omega_baryon,
                _parameters.omega_cb,
                _parameters.omega_nu,
                _parameters.omega_radiation,
                _parameters.f_nu_massless,
                _parameters.f_nu_massive,
                _parameters.w_de,
                _parameters.wa_de);
    }

    void advance()
    {
        std::cout << "Advancing timestepper to step " << _step0 << std::endl;
        // get timestepper up to speed
        for (int step = 0; step < _step0; step++)
            _timestepper->advanceFullStep();

        // we're starting to subcycle after a PM kick
        _timestepper->advanceHalfStep();
    }

    void setupParticles(const int input_flag, const std::string& input_filename)
    {
        const float min_alive_pos = _parameters.oL;
        const float max_alive_pos = _parameters.rL+_parameters.oL;
        if (input_flag)
        {
            std::cout << "Reading file: " << input_filename << std::endl;
            _particles.readRawData(input_filename);
        }
        else
        {
            std::cout << "Generating synthetic data in range [" << min_alive_pos << "," << max_alive_pos << "] " 
                << "rL=" << _parameters.rL << " oL=" << _parameters.oL << std::endl;
            _particles.generateData(_parameters.np, _parameters.rL, _parameters.oL, MEAN_VEL);
            _particles.convert_phys2grid(_parameters.ng, _parameters.rL, _timestepper->aa());
        }

        _particles.reorder(min_alive_pos, max_alive_pos); // TODO:assumes local extent equals the global extent
        std::cout << "\t" << "xx" << " particles in [" << min_alive_pos << "," << max_alive_pos << "]" << std::endl;
    }

    void subCycle()
    {
        _actions.subCycle(*_timestepper, _parameters.nsub, _parameters.gpscal,
                        _parameters.rmax*_parameters.rmax,
                        _parameters.rsm*_parameters.rsm,
                        _parameters.cm_size, _parameters.oL,
                        _parameters.rL+_parameters.oL);
    }

    Parameters parameters() {return _parameters;}
    aosoa_host_type data() {return _particles.aosoa_host;}
    int num_p() {return _particles.aosoa_host.size();}

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

#endif // SOLVER_H
