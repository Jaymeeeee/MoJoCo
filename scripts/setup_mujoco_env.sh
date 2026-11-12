#!/usr/bin/env bash
# Run from any directory to create or update the shared mujoco environment.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
conda_command="${CONDA_EXE:-conda}"

if [[ ! -f third_party/lerobot/pyproject.toml ]]; then
  echo 'Initialize LeRobot first: git submodule update --init --recursive' >&2
  exit 1
fi

# Clear ROS Python paths for Conda itself and every target-environment command.
run_conda() {
  env -u PYTHONPATH PYTHONNOUSERSITE=1 "$conda_command" "$@"
}

run_python() {
  run_conda run --no-capture-output --name mujoco \
    env -u PYTHONPATH PYTHONNOUSERSITE=1 python "$@"
}

if run_python -c 'import sys' >/dev/null 2>&1; then
  run_conda env update --name mujoco --file environment.yml
else
  run_conda env create --name mujoco --file environment.yml --yes
fi

run_python -m pip install --requirement requirements-lock.txt
run_python -m pip install --no-deps --no-build-isolation \
  --editable 'third_party/lerobot[training,diffusion]' --editable '.[dev]'
run_python -m pip check
run_python -c '
import sys
import accelerate
import lerobot
import mujoco
import torch
import trackio
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from lerobot.scripts.lerobot_train import main

print({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mujoco": mujoco.__version__,
    "lerobot": lerobot.__file__,
    "training_imports": "ok",
})
'
