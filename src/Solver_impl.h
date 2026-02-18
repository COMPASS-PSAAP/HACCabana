#include "Solver.h"

namespace HACCabana
{

template <class MemorySpace, class ExecutionSpace>
Solver<MemorySpace, ExecutionSpace>::Solver( const int step0 )
    : _step0(step0)
    , _parameters()
    , _particles()
    , _actions( &_particles )
{
    
}

template <class MemorySpace, class ExecutionSpace>
void Solver<MemorySpace, ExecutionSpace>::setup(const int config_flag, const std::string& configuration_filename)
{
    if (config_flag)
    {
        _parameters.load_from_file(configuration_filename);
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

template <class MemorySpace, class ExecutionSpace>
void Solver<MemorySpace, ExecutionSpace>::advance()
{
    std::cout << "Advancing timestepper to step " << _step0 << std::endl;
    // get timestepper up to speed
    for (int step = 0; step < _step0; step++)
        _timestepper->advanceFullStep();

    // we're starting to subcycle after a PM kick
    _timestepper->advanceHalfStep();
}
template <class MemorySpace, class ExecutionSpace>
void Solver<MemorySpace, ExecutionSpace>::setupParticles(const int input_flag, const std::string& input_filename)
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
    std::cout << "\t" << _particles.end-_particles.begin << " particles in [" << min_alive_pos << "," << max_alive_pos << "]" << std::endl;
}

template <class MemorySpace, class ExecutionSpace>
void Solver<MemorySpace, ExecutionSpace>::subCycle()
{
    _actions.subCycle(*_timestepper, _parameters.nsub, _parameters.gpscal,
                    _parameters.rmax*_parameters.rmax,
                    _parameters.rsm*_parameters.rsm,
                    _parameters.cm_size, _parameters.oL,
                    _parameters.rL+_parameters.oL);
}

} // end namespace HACCabana