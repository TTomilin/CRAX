# Final Batch Jobs

This directory contains batch job scripts for the final training runs after fixing the Lagrangian implementation bug.

## Summary

Total: **10 batch jobs** across 3 environments

### Point Goal (3 runs, 20 seeds each)
- `pointgoal_ppol_sb0.05.sh` - PPO-Lagrange, safety bound 0.05
- `pointgoal_ppol_sb0.075.sh` - PPO-Lagrange, safety bound 0.075
- `pointgoal_ppol_sb0.025.sh` - PPO-Lagrange, safety bound 0.025

### Ant Velocity (4 runs, 20 seeds each)
- `ant_velocity_ppo_sb0.1.sh` - PPO, safety bound 0.1, max velocity 3.0
- `ant_velocity_ppol_sb0.1.sh` - PPO-Lagrange, safety bound 0.1, max velocity 3.0
- `ant_velocity_ppol_sb0.2.sh` - PPO-Lagrange, safety bound 0.2, max velocity 3.0
- `ant_velocity_ppol_sb0.3.sh` - PPO-Lagrange, safety bound 0.3, max velocity 3.0

### Humanoid Hop (3 runs, 10 seeds each)
- `humanoid_hop_ppol_sb0.006_rscale0.1.sh` - PPO-Lagrange, safety bound 0.006, reward scaling 0.1
- `humanoid_hop_ppol_sb0.006_rscale0.01.sh` - PPO-Lagrange, safety bound 0.006, reward scaling 0.01
- `humanoid_hop_ppol_sb0.006_rscale0.001.sh` - PPO-Lagrange, safety bound 0.006, reward scaling 0.001

## Configuration Files

All configs are in `configs/final/` directory. Each config:
- Uses wandb project suffixed with `_final`:
  - `safe-brax-pointgoal_final`
  - `safe-brax-ant_velocity_final`
  - `safe-brax-humanoid_hop_final`
- Includes video recording settings (videos generated automatically)
- Seeds are specified via CLI args in batch jobs

## Usage

Submit a job:
```bash
sbatch final_batch_jobs/pointgoal_ppol_sb0.05.sh
```

Submit all jobs:
```bash
for script in final_batch_jobs/*.sh; do
    sbatch "$script"
done
```

## Output Locations

- Logs: `logs/` directory
- Videos: `videos/` directory (generated automatically after training)
- Metrics: `runs/final_experiments/` directory
- Wandb: Projects suffixed with `_final`

## Notes

- All jobs request 1 GPU
- Videos are generated automatically (no need for `--skip-video` flag)
- Rollout evaluation is performed automatically after training
- Each job runs multiple seeds sequentially



