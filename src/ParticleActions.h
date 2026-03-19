#ifndef PARTICLE_ACTIONS_H
#define PARTICLE_ACTIONS_H

#include <string>

#include <Cabana_Core.hpp>

#include "Definitions.h"
#include "TimeStepper.h"
#include "Particles.h"
#include "ForceSolvers.h"

#include <sys/time.h>

#ifdef HACCabana_ENABLE_CANOPY
template<class AoSoAType, class Field>
using DefaultForceSolverType = CanopyForceSolver<AoSoAType, Field>;
#else
template<class AoSoAType, class Field>
using DefaultForceSolverType = P3MForceSolver<AoSoAType, Field>;
#endif

double mytime() {
  timeval tv;
  gettimeofday(&tv, NULL);
  double time = 1.0*tv.tv_sec;
  time += tv.tv_usec*1.e-6;
  return time;
}

namespace HACCabana 
{

template <class ParticleType, template<class, class> class ForceSolverType = DefaultForceSolverType>
class ParticleActions
{
    public:
    using particle_type = ParticleType;
    using memory_space = typename particle_type::memory_space;
    using execution_space = typename particle_type::execution_space;
    using Field = typename particle_type::Field;
    using aosoa_type = typename particle_type::aosoa_type;
    using aosoa_host_type = typename particle_type::aosoa_host_type;
    using force_solver_type = ForceSolverType<aosoa_type, Field>;

    ParticleActions() {}

    ParticleActions(particle_type *P_)
        : P(P_)
    {
    };

    ~ParticleActions() {}

    void setParticles(particle_type *P_)
    {
        P = P_;
    }

    void subCycle(TimeStepper &ts, const int nsub, const float gpscal, const float rmax2, const float rsm2, 
        const float cm_size, const float min_pos, const float max_pos)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // copy particles to GPU
        aosoa_type aosoa_device("aosoa_device", P->num_p);
        Cabana::deep_copy(aosoa_device, P->aosoa_host);

        // ------------------------------------------------------------------------------------
        const double stepFraction = 1.0/nsub;

        // mass convertion from IC np units to ng grid units. Required since mass = 1 in Gravity only HACC. 
        const float divscal = gpscal*gpscal*gpscal;
        const float pi = 4.0*atanf(1.0);
        // c = G*dt/dy*1/a in code units. 
        // a = Gm/x^2. 
        // G = (3/2)Omega_m *(1/4pi) 
        // in code units, dt/dy for time transformation, 1/a since dp/dy = -gradphi/a.
        const float c = divscal/4.0/pi*ts.fscal()*ts.tau()*stepFraction;

        const float pf = powf(ts.pp(), (1.0 + 1.0 / ts.alpha()));
        const float prefactor = 1.0 / (ts.alpha() * ts.adot() * pf);
        float tau = ts.tau()*stepFraction;

        _force_solver.setup_subcycle(aosoa_device,
                      P->begin, P->end, c, cm_size,
                      min_pos, max_pos,
                      rmax2, rsm2);

        // create the cell list on the GPU
        // NOTE: fuzz particles (outside of overload) are not included
        // float dx = cm_size;
        // float x_min = min_pos;
        // float x_max = max_pos;

        // float grid_delta[3] = {dx, dx, dx};
        // float grid_min[3] = {x_min, x_min, x_min};
        // float grid_max[3] = {x_max, x_max, x_max};

        // auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
        // auto cell_list = Cabana::createLinkedCellList(
        //         position, P->begin, P->end, grid_delta, grid_min, grid_max );
        // Cabana::permute(cell_list, aosoa_device);
        // Kokkos::fence();

        double kick_time = 0.0f;
        // SKS subcycles
        for(int step=0; step < nsub; ++step) 
        {
            std::cout << "Doing substep " << step << std::endl;

            //half stream
            this->updatePos(aosoa_device, prefactor*tau*0.5);

            // kick
            double tmp = mytime();
            _force_solver.updateVel(aosoa_device);
            kick_time += mytime() - tmp;

            // auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
            // printf("R%d: step%d: aosoa size: %d\n", rank, step, aosoa_device.size());
            // for (std::size_t i = 0; i < aosoa_device.size(); i++)
            // {
            // printf("R%d: step%d p(%.2lf, %.2lf, %.2lf)\n", rank, step,
            //     position(i, 0),position(i, 1), position(i, 2));
            // }

            //half stream
            this->updatePos(aosoa_device, prefactor*tau*0.5);
        }

        std::cout << "kick time " << kick_time << std::endl;

        // copy GPU particles back to host
        P->aosoa_host.resize(aosoa_device.size());
        Cabana::deep_copy(P->aosoa_host, aosoa_device);
    }

    void updatePos(aosoa_type aosoa_device, float prefactor)
    {
        auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_device, "velocity");

        Kokkos::parallel_for("stream", Kokkos::RangePolicy<execution_space>(0, P->num_p),
        KOKKOS_LAMBDA(const int i) {
            position(i,0) = position(i,0) + prefactor * velocity(i,0);
            position(i,1) = position(i,1) + prefactor * velocity(i,1);
            position(i,2) = position(i,2) + prefactor * velocity(i,2);
        });
        Kokkos::fence();
    }

    template <class CellListType>
    void updateVel(aosoa_type aosoa_device, CellListType cell_list,
        const float c, const float rmax2, const float rsm2)
    {
        auto position = Cabana::slice<Field::Position>(aosoa_device, "position");
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_device, "velocity");
        auto bin_index = Cabana::slice<Field::BinIndex>(aosoa_device, "bin_index");

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

        Cabana::SimdPolicy<VECTOR_LENGTH, execution_space> simd_policy(P->begin, P->end);
        Cabana::simd_parallel_for( simd_policy, vector_kick, "kick" ); 

        Kokkos::fence();
    }


    private:
    particle_type *P;
    force_solver_type _force_solver;
};

} // end namespace HACCabana


#endif
