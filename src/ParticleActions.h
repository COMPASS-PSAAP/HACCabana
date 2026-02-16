#ifndef PARTICLE_ACTIONS_H
#define PARTICLE_ACTIONS_H

#include <string>

#include <Cabana_Core.hpp>

#include "Definitions.h"
#include "TimeStepper.h"
#include "Particles.h"

namespace HACCabana 
{
  template <class ParticleType>
  class ParticleActions
  {
  public:
    using memory_space = typename ParticleType::memory_space;
    using execution_space = typename ParticleType::execution_space;
    using fields = typename ParticleType::field;
    using aosoa_type = typename ParticleType::aosoa_type;
    using aosoa_host_type = typename ParticleType::aosoa_host_type;

    ParticleActions();
    ParticleActions(ParticleType *P_);
    ~ParticleActions();
    
    void setParticles(ParticleType *P_);
    void subCycle(TimeStepper &ts, const int nsub, const float gpscal, const float rmax2, const float rsm2,\
                  const float cm_size, const float min_pos, const float max_pos);
    void updatePos(aosoa_type aosoa_device, float prefactor);
    void updateVel(aosoa_type aosoa_device, Cabana::LinkedCellList<memory_space> cell_list,
                   const float c, const float rmax2, const float rsm2);
  

  private:
    ParticleType *P;
  };
}
#endif
