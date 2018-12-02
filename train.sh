#!/bin/sh
rm -f log.txt
rm -r -f log/*
rm -f model/*
rm -f parameters
module purge
module load StdEnv gcc/4.8.5 python_gpu/3.6.1
bsub -W 24:00 -R "rusage[mem=10240, ngpus_excl_p=1]" -R "select[gpu_model1==GeForceGTX1080Ti]" python main.py --option=train_test

