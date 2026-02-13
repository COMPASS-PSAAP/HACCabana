#include "Solver.h"

namespace HACCabana
{

template <class MemorySpace, class ExecutionSpace>
Solver<MemorySpace, ExecutionSpace>( int config_flag )
    : _particles()
    , _actions( &_particles )
{
    if (config_flag)
    {
        Params.load_from_file(configuration_filename);
    }

    TimeStepper ts(
        Params.alpha,
            Params.a_in,
            Params.a_fin,
            Params.nsteps,
            Params.omega_matter,
            Params.omega_cdm,
            Params.omega_baryon,
            Params.omega_cb,
            Params.omega_nu,
            Params.omega_radiation,
            Params.f_nu_massless,
            Params.f_nu_massive,
            Params.w_de,
        Params.wa_de);
}

} // end namespace HACCabana