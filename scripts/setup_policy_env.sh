#!/usr/bin/env bash
# Run from any directory. Existing collection/conversion environments are untouched.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
conda_command="${CONDA_EXE:-conda}"
if ! "$conda_command" run --name mujoco_policy python -c 'import sys' >/dev/null 2>&1; then
  "$conda_command" env create --file environment-policy.yml --yes
fi
"$conda_command" run --no-capture-output --name mujoco_policy \
  env -u PYTHONPATH python -m pip install --requirement requirements-policy-lock.txt
"$conda_command" run --no-capture-output --name mujoco_policy \
  env -u PYTHONPATH python -m pip install --no-deps --no-build-isolation \
  --editable third_party/lerobot --editable .
"$conda_command" run --no-capture-output --name mujoco_policy \
  env -u PYTHONPATH python -m pip check
"$conda_command" run --no-capture-output --name mujoco_policy \
  env -u PYTHONPATH python -c \
  'import torch, mujoco, lerobot; from lerobot.policies.act.modeling_act import ACTPolicy; print({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "mujoco": mujoco.__version__, "lerobot": lerobot.__file__})'
