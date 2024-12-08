#!/bin/bash

# Base repo is in scratch because it makes data transfers easier
export PROJ_HOME=$SCRATCH
echo "PROJ_HOME is $PROJ_HOME"

module purge
module load anaconda/2021.05-py38
module load modtree/gpu
module load gcc/11.2.0
module load cmake/3.20.0
module load cuda/12.0.1
module load python/3.9.5
module list

export SPLAT_HOME=$PROJ_HOME/<base-directory>
export SPLAT_DATASET_PATH=$SPLAT_HOME/data

export OMP_NUM_THREADS=32
export RAFT_HOME=$PROJ_HOME/raft/cpp
export OPENBLAS_HOME=$PROJ_HOME/openblas
export LIB64=/usr/lib64

export CUDA_HOME=$(dirname $(which nvcc))/../

# For x86(-64) architecture only
export TCNN_CUDA_ARCHITECTURES=80

