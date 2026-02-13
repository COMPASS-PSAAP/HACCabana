#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define POLY_ORDER 5
#define MEAN_VEL 10000.0
#define MAX_ERR 1.0e-06f

// This might need to be changed for AMD and Intel GPUs. Nvidia warp size is 32. 
#define VECTOR_LENGTH 64

#endif
