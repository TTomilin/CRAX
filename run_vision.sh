#!/bin/bash

algos=("ppo")
#algos=("ppo" "ppo_cost" "ppo_lag" "ppo_pid" "ppo_saute" "p3o" "focops")
#envs=("safe_height")
envs=("safe_goal_point")
#envs=("safe_walker")
#envs=("safe_spider")
#envs=("safe_push")
#envs=("safe_point_circle")
safety_bounds=(25.0)
lambdas=(3.0)
seeds=(1)
#seeds=(1 2 3 4 5)
levels=(1)
#levels=(1 2 3)
penalties=(-1.0)

for alg in "${algos[@]}"; do
    for env_name in "${envs[@]}"; do
        for safety_bound in "${safety_bounds[@]}"; do
            for lambda in "${lambdas[@]}"; do
                for penalty in "${penalties[@]}"; do
                    for level in "${levels[@]}"; do
                        for seed in "${seeds[@]}"; do
                            echo "Submitting job for combination: $alg, env=$env_name, Level=$level, Bound=$safety_bound, Lambda=$lambda, Penalty=$penalty"

                            # Create an SBATCH script
                            cat <<EOF | sbatch
#!/bin/bash
#SBATCH -p gpu_h100
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --time 5:00:00
#SBATCH --gres gpu:1
#SBATCH --job-name=brax
#SBATCH -o /home/mbeurskens/projects/slurm/%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate crax
export MUJOCO_GL=egl
cd /gpfs/home6/mbeurskens/projects/CRAX/
PYTHONPATH=$HOME/CRAX python3 ~/projects/CRAX/train_env.py \
        --alg $alg \
        --env_name $env_name \
        --safety_bound $safety_bound \
        --lagrangian_coef_rate $lambda \
        --difficulty $level \
        --seeds $seed \
        --saute-violation-penalty $penalty \
        --cost-weight 1.0 \
        --cameras "track" \
        --kappa_increase_factor 1.1 \
        --nu_lr 1.0 \
        --pid_kp 10.0 \
        --pid_ki 0.01 \
        --pid_kd 0.01 \
        --pid_integral_clip 1 \
        --pid_deriv_ema_beta 0.95 \
        --skip_rollout \
        --skip_video \
        --video_fps 50 \
        --num_evals 0 \
        --num_timesteps 1e4 \
        --wandb_tags ICML LEVEL_$level \
        --vision \
        --num_envs 64 \
        --vision_render_workers 8 \
        --vision_height 64 \
        --vision_width 64
conda deactivate
EOF
                            sleep 0.1
                        done
                    done
                done
            done
        done
    done
done
