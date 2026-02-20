#ifndef FORCE_SOLVERS_H
#define FORCE_SOLVERS_H

#include <Cabana_Core.hpp>

// Polynomial long range force calculation
KOKKOS_INLINE_FUNCTION
float FGridEvalPoly(float r2)
{
#if POLY_ORDER == 6
  return (0.271431f + r2*(-0.0783394f + r2*(0.0133122f + r2*(-0.00159485f + r2*(0.000132336f + r2*(-0.00000663394f + r2*0.000000147305f))))));
#elif POLY_ORDER == 5
  return (0.269327f + r2*(-0.0750978f + r2*(0.0114808f + r2*(-0.00109313f + r2*(0.0000605491f + r2*-0.00000147177f)))));
#elif POLY_ORDER == 4
  return (0.263729f + r2*(-0.0686285f + r2*(0.00882248f + r2*(-0.000592487f + r2*0.0000164622f))));
#else
  return 0.0f;
#endif
}

template <class AoSoAType, class Field>
class P3MForceSolver
{
  private:
  // Variables needed for P3M
  using memory_space = typename AoSoAType::memory_space;
  using execution_space = typename AoSoAType::execution_space;
  Cabana::LinkedCellList<memory_space, float, 3> _cell_list;
  
  size_t _begin, _end;
  float _rmax2;
  float _rsm2;

  public:
  P3MForceSolver() {}
  ~P3MForceSolver() {}

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

    auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
    _cell_list = Cabana::createLinkedCellList(
            position, _begin, _end, grid_delta, grid_min, grid_max );
    Cabana::permute(_cell_list, aosoa_device);
    Kokkos::fence();
  }

  void updateVel(AoSoAType& aosoa_device, const float c, float rmax2, float rsm2)
  {
    auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
    auto velocity = Cabana::slice<Field::Velocity>(aosoa_device, "velocity");
    auto force = Cabana::slice<Field::Force>(aosoa_device, "force");
    auto bin_index = Cabana::slice<Field::BinIndex>(aosoa_device, "bin_index");
    auto cell_list = _cell_list;

    Kokkos::parallel_for("copy_bin_index", Kokkos::RangePolicy<execution_space>(0, cell_list.totalBins()),
    KOKKOS_LAMBDA(const int i)
    {
        int bin_ijk[3];
        cell_list.ijkBinIndex(i, bin_ijk[0], bin_ijk[1], bin_ijk[2]);
        for (size_t ii = cell_list.binOffset(bin_ijk[0], bin_ijk[1], bin_ijk[2]); 
            ii < cell_list.binOffset(bin_ijk[0], bin_ijk[1], bin_ijk[2]) +
            cell_list.binSize(bin_ijk[0], bin_ijk[1], bin_ijk[2]); 
            ++ii)
        bin_index(ii) = i;
    });
    Kokkos::fence();

    auto vector_kick = KOKKOS_LAMBDA(const int s, const int a)
    {
        int bin_ijk[3];
        cell_list.ijkBinIndex(bin_index.access(s,a), bin_ijk[0], bin_ijk[1], bin_ijk[2]);

        // Zero force
        force.access(s,a,0)  = 0.0;
        force.access(s,a,1)  = 0.0;
        force.access(s,a,2)  = 0.0;

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
                    force.access(s,a,0) += dx * tmp;
                    force.access(s,a,1) += dy * tmp;
                    force.access(s,a,2) += dz * tmp;
                    }
                }
            }
        }
        }
        velocity.access(s,a,0) += force.access(s,a,0) * c;
        velocity.access(s,a,1) += force.access(s,a,1) * c;
        velocity.access(s,a,2) += force.access(s,a,2) * c;
    };

    Cabana::SimdPolicy<VECTOR_LENGTH, execution_space> simd_policy(_begin, _end);
    Cabana::simd_parallel_for( simd_policy, vector_kick, "kick" ); 

    Kokkos::fence();
  }
};
#ifdef HACCabana_ENABLE_CANOPY
#include "CanopyForceSolver.h"
#endif

#endif