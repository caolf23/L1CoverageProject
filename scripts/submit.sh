#!/usr/bin/env bash
#SBATCH --job-name=eval_codex_w
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=a01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/fit/alex/LufanCao/codex_ppo/logs/eval_codex_w_%j.out
#SBATCH --error=/home/fit/alex/LufanCao/codex_ppo/logs/eval_codex_w_%j.err

set -euo pipefail

PROJECT_DIR="/home/fit/alex/LufanCao/codex_ppo"
SCRIPT_PATH="$PROJECT_DIR/scripts/eval_codex_w.py"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Force real-time logging from Python and subprocess stdio.
export PYTHONUNBUFFERED=1

echo "[$(date '+%F %T')] Starting eval_codex_w.py"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"

srun --unbuffered stdbuf -oL -eL python3 -u "$SCRIPT_PATH"

echo "[$(date '+%F %T')] Finished eval_codex_w.py"
