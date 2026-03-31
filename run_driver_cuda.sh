#!/bin/bash
# example using synthetic data at step 0
OMP_PROC_BIND=false drivers/HACCabana_Driver_CUDA -d -t 0 -c ../064.indat.params -p 80 -s 3 -f fmm
