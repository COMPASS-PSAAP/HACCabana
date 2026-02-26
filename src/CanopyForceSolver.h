#ifndef CANOPY_FORCE_SOLVER_H
#define CANOPY_FORCE_SOLVER_H

#include <Cabana_Core.hpp>

template <class AoSoAType, class Field>
class CanopyForceSolver
{
  private:
  // Variables needed for Canopy
  using memory_space = typename AoSoAType::memory_space;
  using execution_space = typename AoSoAType::execution_space;
  using MD = Canopy::ParticleMetadata<AoSoAType, Field::Position, Field::Gravity, Field::Potential, Field::Force>; 
  static constexpr num_coefficients = 6;
  std::shared_ptr<Canopy::Solver<execution_space, memory_space, MD,
                                 2, num_coefficients>> _solver;
  
  size_t _begin, _end;
  float _rmax2;
  float _rsm2;

  public:
  CanopyForceSolver() {}
  ~CanopyForceSolver() {}

  void setup_subcycle(AoSoAType& aosoa_device,
                      const size_t begin, const size_t end,
                      const float c, const float cm_size,
                      const float min_pos, const float max_pos,
                      const float rmax2, const float rsm2)
  {
    _begin = begin;
    _end = _end;
    _rmax2 = rmax2;
    _rsm2 = rsm2;
    
    float dx = cm_size;
    float x_min = min_pos;
    float x_max = max_pos;

    float grid_delta[3] = {dx, dx, dx};
    float grid_min[3] = {x_min, x_min, x_min};
    float grid_max[3] = {x_max, x_max, x_max};

    _solver = Canopy::createSolver<execution_space, memory_space, MD, 2, num_coefficients>
  }

  void updateVel(AoSoAType& aosoa_device, const float c, float rmax2, float rsm2)
  {
    _solver->solve(aosoa_device, 1);
    
    auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
    auto velocity = Cabana::slice<Field::Velocity>(aosoa_device, "velocity");

    auto vector_kick = KOKKOS_LAMBDA(const int s, const int a)
    {
        int bin_ijk[3];
        cell_list.ijkBinIndex(bin_index.access(s,a), bin_ijk[0], bin_ijk[1], bin_ijk[2]);

        float force[3] = {0.0, 0.0, 0.0};
        for (int ii=-1; ii<2; ++ii) 
        {
        if (bin_ijk[0] + ii < 0 || bin_ijk[0] + ii >= cell_list.numBin(0))
            continue;
        for (int jj=-1; jj<2; ++jj) 
        {
            if (bin_ijk[1] + jj < 0 || bin_ijk[1] + jj >= cell_list.numBin(1))
                continue;
            for (int kk=-1; kk<2; ++kk) 
            {
                if (bin_ijk[2] + kk < 0 || bin_ijk[2] + kk >= cell_list.numBin(2))
                    continue;

                const size_t binOffset = cell_list.binOffset(bin_ijk[0] + ii, bin_ijk[1] + jj, bin_ijk[2] + kk);
                const int binSize = cell_list.binSize(bin_ijk[0] + ii, bin_ijk[1] + jj, bin_ijk[2] + kk);

                for (int j = binOffset; j < binOffset+binSize; ++j) 
                {
                    const float dx = position(j,0)-position.access(s,a,0);
                    const float dy = position(j,1)-position.access(s,a,1);
                    const float dz = position(j,2)-position.access(s,a,2);
                    const float dist2 = dx * dx + dy * dy + dz * dz;
                    if (dist2 < rmax2) 
                    {
                    const float dist2Err = dist2 + rsm2;
                    const float tmp =  1.0f/Kokkos::sqrt(dist2Err*dist2Err*dist2Err) - FGridEvalPoly(dist2);
                    force[0] += dx * tmp;
                    force[1] += dy * tmp;
                    force[2] += dz * tmp;
                    }
                }
            }
        }
        }
        velocity.access(s,a,0) += force[0] * c;
        velocity.access(s,a,1) += force[1] * c;
        velocity.access(s,a,2) += force[2] * c;
    };

    Cabana::SimdPolicy<VECTOR_LENGTH, execution_space> simd_policy(_begin, _end);
    Cabana::simd_parallel_for( simd_policy, vector_kick, "kick" ); 

    Kokkos::fence();
  }
};

#endif // CANOPY_FORCE_SOLVER_H
