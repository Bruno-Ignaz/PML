#!/bin/bash
# Snapshot the current code and submit one Slurm job per dataset.
#
#   ./run.sh RUN_NAME [drifting.train arguments]
#   ./run.sh t0.05 --temperature 0.05
#   OUTPUT_ROOT=~/outputs ./run.sh t0.05      # other output location
#
# Creates OUTPUT_ROOT/RUN_NAME/ (default: outputs/ next to this script):
#   code/        the code as it is now (no .venv or other ignored files),
#                plus git.txt (commit and changes) and env.txt (environment variables)
#   data/        results, one folder per dataset
#   slurm_logs/  one log per dataset
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME=${1:?usage: ./run.sh RUN_NAME [drifting.train arguments]}
shift
RUN_DIR="${OUTPUT_ROOT:-outputs}/$RUN_NAME"
PYTHON="$PWD/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "No .venv found, run 'uv sync' first." >&2; exit 1; }
[[ ! -e "$RUN_DIR" ]] || { echo "$RUN_DIR already exists." >&2; exit 1; }
mkdir -p "$RUN_DIR"/{code,data,slurm_logs}
RUN_DIR="$(cd "$RUN_DIR" && pwd)"  # absolute, for Slurm

# Tracked and new files, without anything in .gitignore
git ls-files -z --cached --others --exclude-standard \
  | tar --null -T - -cf - | tar -xf - -C "$RUN_DIR/code"
{ git rev-parse HEAD; git status --short; } > "$RUN_DIR/code/git.txt"
env | sort > "$RUN_DIR/code/env.txt"

sbatch \
  --job-name="$RUN_NAME" \
  --chdir="$RUN_DIR/code" \
  --output="$RUN_DIR/slurm_logs/%A_%a.out" \
  "$RUN_DIR/code/slurm/train.sbatch" "$PYTHON" "$RUN_DIR/data" "$@"

echo "Run folder: $RUN_DIR"
