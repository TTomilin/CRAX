#!/bin/bash

#SBATCH --job-name=ant_velocity_ppol_sb0.3
#SBATCH --output=logs/ant_velocity_ppol_sb0.3_%j.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus=1

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.8.0 cuDNN/9.10.1.4-CUDA-12.8.0

# Activate virtual environment
source ~/venvs/safebrax/bin/activate

# Set environment variables for JAX/GPU and headless rendering
export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:$EBROOTCUDNN/lib:$LD_LIBRARY_PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORM_NAME=gpu
export XLA_FLAGS="--xla_gpu_triton_gemm_any=True"
export MUJOCO_GL=egl

# Change to project directory
cd /home/20191873/safebrax

# Create logs and videos directories if they don't exist
mkdir -p logs
mkdir -p videos

echo "=================================================="
echo "Training Ant Velocity PPO-L SB 0.3 (20 seeds)"
echo "Started at $(date)"
echo "=================================================="

# Run training with 20 seeds
python train_from_config.py \
    --config configs/final/ant_velocity_ppol_sb0.3.json \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

echo "=================================================="
echo "Training completed at $(date)"
echo "=================================================="



