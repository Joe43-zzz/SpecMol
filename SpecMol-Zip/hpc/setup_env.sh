#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-specmol}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
PYG_VERSION="${PYG_VERSION:-2.6.1}"
PYG_WHEEL_TAG="${PYG_WHEEL_TAG:-torch-2.4.0+cu124}"

if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda was not found. Load/install conda on the HPC before running this script." >&2
    exit 1
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

conda install -y -c pytorch -c nvidia \
  "pytorch=$TORCH_VERSION" \
  pytorch-cuda=12.4

# PyTorch 2.4.x can fail to import with MKL 2025 on some HPC images:
# libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
conda install -y "mkl<2025" "intel-openmp<2025"

conda install -y -c conda-forge \
  rdkit \
  pandas \
  numpy \
  scipy \
  scikit-learn \
  networkx \
  seaborn \
  tqdm

python -m pip install \
  "torch_geometric==$PYG_VERSION" \
  pyg_lib \
  torch_scatter \
  torch_sparse \
  torch_cluster \
  torch_spline_conv \
  -f "https://data.pyg.org/whl/${PYG_WHEEL_TAG}.html"

python -m pip install deepchem
python -m pip install unimol_tools
python -m pip install --force-reinstall "numpy==1.26.4" "scipy==1.11.4"

python - <<'PY'
import deepchem
import networkx
import rdkit
import sklearn
import torch
import torch_geometric

print("Environment import check passed")
print("torch", torch.__version__)
print("torch_geometric", torch_geometric.__version__)
print("deepchem", deepchem.__version__)
print("cuda_available_requires_slurm_gpu_job", torch.cuda.is_available())
PY
