#!/bin/bash

mdlf='ccsn_model.dat'
ccsnf='all_models_w_time.hdf5'
hydrof='21out.bin'
savef='21_test_dust_.hdf5'

python fryhy_to_sndust.py $mdlf $ccsnf $hydrof $savef 21