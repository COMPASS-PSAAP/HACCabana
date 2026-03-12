#!/bin/bash
# example using synthetic data at step 0
OMP_PROC_BIND=false drivers/HACCabana_Driver_SERIAL -s -t 0 -c ../064.indat.params -p 500
