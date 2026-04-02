#ifndef PARTICLES_H
#define PARTICLES_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>

#include <Cabana_Core.hpp>

#include "HACCabana_Definitions.h"

namespace HACCabana
{

template <class MemorySpace, class ExecutionSpace>
class Particles
{
public:
    struct ParticleData
    {
    using member_types = Cabana::MemberTypes<int64_t, float[3], float[3], float[3], float, float, int>;

        struct Field
        {
            enum : int { ParticleID = 0, Position = 1, Velocity = 2, Force = 3, Gravity = 4, Potential = 5, BinIndex = 6 };
        };
    };
    using memory_space = MemorySpace;
    using execution_space = ExecutionSpace;
    using member_types = typename ParticleData::member_types;
    using Field = typename ParticleData::Field;
    using aosoa_type = Cabana::AoSoA<member_types, memory_space, VECTOR_LENGTH>;
    using aosoa_host_type = Cabana::AoSoA<member_types, Kokkos::HostSpace, VECTOR_LENGTH>;

    aosoa_host_type aosoa_host;

    Particles() {}

    ~Particles() {}


    void convert_phys2grid(int ng, float rL, float a)
        {
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_host, "velocity");

        const float phys2grid_pos = ng/rL;
        const float phys2grid_vel = phys2grid_pos/100.0;
        const float scaling = phys2grid_vel*a*a;
        for (int i=0; i<aosoa_host.size(); ++i)
        {
            for (int j=0; j<3; ++j) {
            velocity(i,j) *= scaling;
            }
        }
    }

    void generateData(const int np, const float rl, const float ol,
                      const float mean_vel, const bool weak_scaling = false,
                      const std::size_t weak_scaling_num_particles = 0)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const std::size_t default_num_particles =
            static_cast<std::size_t>(np) * np * np;
        const std::size_t num_p =
            weak_scaling
                ? (weak_scaling_num_particles > 0 ? weak_scaling_num_particles
                                                  : default_num_particles)
                : (rank == 0 ? default_num_particles : 0);
        aosoa_host = aosoa_host_type("aosoa_host", num_p);
        if (num_p == 0)
            return;

        auto id = Cabana::slice<Field::ParticleID>(aosoa_host, "id");
        auto position = Cabana::slice<Field::Position>(aosoa_host, "position");

        const int grid_particles_per_dim =
            (weak_scaling && weak_scaling_num_particles > 0)
                ? std::max(1, static_cast<int>(std::ceil(
                      std::cbrt(static_cast<double>(weak_scaling_num_particles)))))
                : np;
        const float delta = rl/grid_particles_per_dim;  // inter-particle spacing

        // generate data from a grid and offset from a normal random distribution
        // std::random_device rd{};
        std::mt19937 gen{static_cast<std::mt19937::result_type>(rank)};
        std::normal_distribution<float> d1{0.0, 0.05}; // mu=0.0 sigma=0.05

        float min_pos[3];
        float max_pos[3];

        const int64_t particle_id_offset =
            weak_scaling ? static_cast<int64_t>(rank) *
                               static_cast<int64_t>(num_p)
                         : 0;

        for (std::size_t i = 0; i < num_p; ++i)
        {
            id(i) = particle_id_offset + static_cast<int64_t>(i);
            position(i,0) =
                static_cast<float>(i % grid_particles_per_dim) * delta +
                delta*0.5 + d1(gen) + ol;
            min_pos[0] = i==0 ? position(i,0) : min_pos[0] > position(i,0) ? position(i,0) : min_pos[0]; 
            max_pos[0] = i==0 ? position(i,0) : max_pos[0] < position(i,0) ? position(i,0) : max_pos[0]; 

            position(i,1) =
                static_cast<float>((i / grid_particles_per_dim) %
                                   grid_particles_per_dim) *
                delta + delta*0.5 + d1(gen) + ol;
            min_pos[1] = i==0 ? position(i,1) : min_pos[1] > position(i,1) ? position(i,1) : min_pos[1]; 
            max_pos[1] = i==0 ? position(i,1) : max_pos[1] < position(i,1) ? position(i,1) : max_pos[1]; 

            position(i,2) =
                static_cast<float>(
                    i/(grid_particles_per_dim*grid_particles_per_dim)) *
                delta + delta*0.5 + d1(gen) + ol;
            min_pos[2] = i==0 ? position(i,2) : min_pos[2] > position(i,2) ? position(i,2) : min_pos[2];
            max_pos[2] = i==0 ? position(i,2) : max_pos[2] < position(i,2) ? position(i,2) : max_pos[2];
        }

        const float vel_1d = mean_vel/sqrt(3.0);

        std::normal_distribution<float> d2{0.0, 1.0};
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_host, "velocity");
        auto gravity = Cabana::slice<Field::Gravity>(aosoa_host, "gravity");
        auto force = Cabana::slice<Field::Force>(aosoa_host, "force");
        auto potential = Cabana::slice<Field::Potential>(aosoa_host, "potential");
        auto bin_index = Cabana::slice<Field::BinIndex>(aosoa_host, "bin_index");

        for (std::size_t i = 0; i < num_p; ++i)
        {
            for (int j=0; j<3; ++j) {
            velocity(i,j) = vel_1d * d2(gen);
            force(i,j) = 0.0f;
            }
            // Each particle exerts a gravitational pull of the same strength
            gravity(i) = 1.0;
            potential(i) = 0.0f;
            bin_index(i) = 0;
        }

        std::cout << "\t" << num_p << " particles\n" <<
            "\tmin[" << min_pos[0] << "," << min_pos[1] << "," << min_pos[2] << 
            "] max["<< max_pos[0] << "," << max_pos[1] << "," << max_pos[2] << "]" << std::endl;
    }

    void readRawData(std::string file_name) 
    {
        std::ifstream infile(file_name, std::ifstream::binary);

        int num_p;

        // the first int has the number of particles
        infile.read((char*)&num_p, sizeof(int));

        aosoa_host = aosoa_host_type("aosoa_host", num_p);

        if (num_p==0)
            return;

        auto id = Cabana::slice<Field::ParticleID>(aosoa_host, "id");
        auto position = Cabana::slice<Field::Position>(aosoa_host, "position");
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_host, "velocity");
        auto force = Cabana::slice<Field::Force>(aosoa_host, "force");
        auto gravity = Cabana::slice<Field::Gravity>(aosoa_host, "gravity");
        auto potential = Cabana::slice<Field::Potential>(aosoa_host, "potential");
        auto bin_index = Cabana::slice<Field::BinIndex>(aosoa_host, "bin_index");

        float min_pos[3];
        float max_pos[3];

        // id
        for (int i=0; i<num_p; ++i)
            infile.read((char*)&id(i),sizeof(int64_t));
        // pos
        for (int i=0; i<num_p; ++i)
        {
            infile.read((char*)&position(i,0),sizeof(float));
            min_pos[0] = i==0 ? position(i,0) : min_pos[0] > position(i,0) ? position(i,0) : min_pos[0]; 
            max_pos[0] = i==0 ? position(i,0) : max_pos[0] < position(i,0) ? position(i,0) : max_pos[0]; 
        } 
        for (int i=0; i<num_p; ++i)
        {
            infile.read((char*)&position(i,1),sizeof(float));
            min_pos[1] = i==0 ? position(i,1) : min_pos[1] > position(i,1) ? position(i,1) : min_pos[1]; 
            max_pos[1] = i==0 ? position(i,1) : max_pos[1] < position(i,1) ? position(i,1) : max_pos[1]; 
        }
        for (int i=0; i<num_p; ++i)
        {
            infile.read((char*)&position(i,2),sizeof(float));
            min_pos[2] = i==0 ? position(i,2) : min_pos[2] > position(i,2) ? position(i,2) : min_pos[2];
            max_pos[2] = i==0 ? position(i,2) : max_pos[2] < position(i,2) ? position(i,2) : max_pos[2];
        }
        // vel
        for (int i=0; i<num_p; ++i)
            infile.read((char*)&velocity(i,0),sizeof(float));
        for (int i=0; i<num_p; ++i)
            infile.read((char*)&velocity(i,1),sizeof(float));
        for (int i=0; i<num_p; ++i)
            infile.read((char*)&velocity(i,2),sizeof(float));

        for (int i = 0; i < num_p; ++i)
        {
            gravity(i) = 1.0f;
            potential(i) = 0.0f;
            bin_index(i) = 0;
            for (int j = 0; j < 3; ++j)
                force(i,j) = 0.0f;
        }

        infile.close();

        std::cout << "\t" << num_p << " particles\n" << 
            "\tmin[" << min_pos[0] << "," << min_pos[1] << "," << min_pos[2] << "] " << 
            "max["<< max_pos[0] << "," << max_pos[1] << "," << max_pos[2] << "]" << std::endl;
    }

    void reorder(const float min_pos, const float max_pos)
    {
        auto id = Cabana::slice<Field::ParticleID>(aosoa_host, "id");
        auto position = Cabana::slice<Field::Position>(aosoa_host, "position");
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_host, "velocity");
        auto force = Cabana::slice<Field::Force>(aosoa_host, "force");
        auto gravity = Cabana::slice<Field::Gravity>(aosoa_host, "gravity");
        auto potential = Cabana::slice<Field::Potential>(aosoa_host, "potential");
        auto bin_index = Cabana::slice<Field::BinIndex>(aosoa_host, "bin_index");

        std::size_t end = aosoa_host.size();

        // Compact all in-bounds particles into the prefix of the AoSoA and then
        // discard the out-of-bounds suffix so downstream kernels only see live particles.
        std::size_t i = 0;
        while (i < end)
        {
            if (position(i,0) < min_pos || position(i,1) < min_pos || position(i,2) < min_pos ||
                position(i,0) >= max_pos || position(i,1) >= max_pos || position(i,2) >= max_pos)
            {
                const std::size_t tail = end - 1;

                for (int j = 0; j < 3; ++j)
                {
                    float tmp = position(i,j);
                    position(i,j) = position(tail,j);
                    position(tail,j) = tmp;

                    tmp = velocity(i,j);
                    velocity(i,j) = velocity(tail,j);
                    velocity(tail,j) = tmp;

                    tmp = force(i,j);
                    force(i,j) = force(tail,j);
                    force(tail,j) = tmp;
                }

                float tmp_scalar = gravity(i);
                gravity(i) = gravity(tail);
                gravity(tail) = tmp_scalar;

                tmp_scalar = potential(i);
                potential(i) = potential(tail);
                potential(tail) = tmp_scalar;

                int tmp_int = bin_index(i);
                bin_index(i) = bin_index(tail);
                bin_index(tail) = tmp_int;

                int64_t tmp_id = id(i);
                id(i) = id(tail);
                id(tail) = tmp_id;

                --end;
            }
            else
                ++i;
        }

        aosoa_host.resize(end);
    }
};

} // end namespace HACCabana

#endif
