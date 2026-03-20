#ifndef PARTICLES_H
#define PARTICLES_H

#include <string>

#include <Cabana_Core.hpp>

#include "Definitions.h"

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

    void generateData(const int np, const float rl, const float ol, const float mean_vel)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Only rank 0 generate data
        int num_p = 0;
        aosoa_host = aosoa_host_type("aosoa_host", num_p);
        if (rank == 0)
        {
        num_p = np*np*np;
        aosoa_host.resize(num_p);

        auto id = Cabana::slice<Field::ParticleID>(aosoa_host, "id");
        auto position = Cabana::slice<Field::Position>(aosoa_host, "position");

        const float delta = rl/np;  // inter-particle spacing

        // generate data from a grid and offset from a normal random distribution
        // std::random_device rd{};
        std::mt19937 gen{12345};
        std::normal_distribution<float> d1{0.0, 0.05}; // mu=0.0 sigma=0.05

        float min_pos[3];
        float max_pos[3];

        for (int i=0; i<num_p; ++i)
        {
            id(i) = i;
            position(i,0) = (float)(i % np) * delta + delta*0.5 + d1(gen) + ol;
            min_pos[0] = i==0 ? position(i,0) : min_pos[0] > position(i,0) ? position(i,0) : min_pos[0]; 
            max_pos[0] = i==0 ? position(i,0) : max_pos[0] < position(i,0) ? position(i,0) : max_pos[0]; 

            position(i,1) = (float)(i / np % np) * delta + delta*0.5 + d1(gen) + ol;
            min_pos[1] = i==0 ? position(i,1) : min_pos[1] > position(i,1) ? position(i,1) : min_pos[1]; 
            max_pos[1] = i==0 ? position(i,1) : max_pos[1] < position(i,1) ? position(i,1) : max_pos[1]; 

            position(i,2) = (float)(i/(np*np)) * delta + delta*0.5 + d1(gen) + ol;
            min_pos[2] = i==0 ? position(i,2) : min_pos[2] > position(i,2) ? position(i,2) : min_pos[2];
            max_pos[2] = i==0 ? position(i,2) : max_pos[2] < position(i,2) ? position(i,2) : max_pos[2];
        }

        const float vel_1d = mean_vel/sqrt(3.0);

        std::normal_distribution<float> d2{0.0, 1.0};
        auto velocity = Cabana::slice<Field::Velocity>(aosoa_host, "velocity");
        auto gravity = Cabana::slice<Field::Gravity>(aosoa_host, "gravity");

        for (int i=0; i<num_p; ++i)
        {
            for (int j=0; j<3; ++j) {
            velocity(i,j) = vel_1d * d2(gen);
            }
            // Each particle exerts a gravitational pull of the same strength
            gravity(i) = 1.0;
        }

        std::cout << "\t" << num_p << " particles\n" <<
            "\tmin[" << min_pos[0] << "," << min_pos[1] << "," << min_pos[2] << 
            "] max["<< max_pos[0] << "," << max_pos[1] << "," << max_pos[2] << "]" << std::endl;
        }
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

        auto end = aosoa_host.size();

        // Relocate any particle outside of the boundary to the end of the 
        // aosoa -- outside particles start at end until the end of the aosoa.
        for (int i=0; i<end; ++i) {
            if (position(i,0) < min_pos || position(i,1) < min_pos || position(i,2) < min_pos ||
                position(i,0) >= max_pos || position(i,1) >= max_pos || position(i,2) >= max_pos)
            {
            for (int j=0; j<3; ++j)
            {
                float tmp;
                tmp = position(i,j);
                position(i,j) = position(end-1,j);
                position(end-1,j) = tmp;
                tmp = velocity(i,j);
                velocity(i,j) = velocity(end-1,j);
                velocity(end-1,j) = tmp;
            }
            int64_t tmp2 = id(i);
            id(i) = id(end-1);
            id(end-1) = tmp2;

            --end;
            --i;
            }
        }
    }
};

} // end namespace HACCabana

#endif
