#!/usr/bin/env bash
#SBATCH --job-name=qm9mu_dft
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# QOS wall cap is 8h (probed 2026-06-02). Two-phase use to stay under it:
#   build : PREP_ONLY=1 sbatch ...           (prepare+extract+make, exits)
#   arm   : ARM=V0       sbatch ...           (one arm, dedicated GPU -> fast)
# ARM unset + PREP_ONLY unset runs all 7 arms 4-concurrent (legacy; may exceed 8h).

# QM9 dipole(mu) 7-arm pilot on TRUE GDB-9 DFT geometry (Stage-2 deciding bet).
# Mirror of run_qm9mu_pilot.sbatch but: SDF carries true DFT coords (Solution B,
# canonical SMILES atom order) so the 2D graph is IDENTICAL to the MMFF cell and
# only the geometry Uni-Mol sees changes. Chain: prepare_dft (from the validated
# solB cache) -> extract Uni-Mol pair -> make (DFT dirs) -> 7 arms.
# Pre-registered gate (vs RF 0.9249 on the same set + vs MMFF's T8_geom -0.033) is
# applied AFTER collection. Does NOT set CUDA_VISIBLE_DEVICES (Slurm owns it).
set -o pipefail
ENV_NAME="${ENV_NAME:-specmol}"
cd "$HOME/zhoutianyang/SpecMol/SpecMol-Zip"
mkdir -p hpc/logs hpc/results pkl

if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)";
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then source "$HOME/miniconda3/etc/profile.d/conda.sh";
else source "$HOME/anaconda3/etc/profile.d/conda.sh"; fi
conda activate "$ENV_NAME"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
CKPT="$HOME/miniconda3/envs/$ENV_NAME/lib/python3.10/site-packages/unimol_tools/weights/mol_pre_all_h_220816.pt"
DICT="$HOME/miniconda3/envs/$ENV_NAME/lib/python3.10/site-packages/unimol_tools/weights/mol.dict.txt"
echo "node=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  $(date)"

export QM9MU_UNIMOL_OUT=unimol_out_qm9mu_dft
export QM9MU_OUTPUT_DIR=down_task_qm9mu_unimol_v2_dft
DATA=down_task_qm9mu_unimol_v2_dft/processed/qm9mu_all.pt
if [ ! -f "$DATA" ]; then
  [ -f data/qm9/qm9_dft_solB_cache.pkl ] || { echo "MISSING data/qm9/qm9_dft_solB_cache.pkl (scp from local)"; exit 1; }
  echo "[step1] prepare_qm9mu_dft (DFT coords from solB cache)"; python -u prepare_qm9mu_dft_input.py
  echo "[step2] extract Uni-Mol pair (DFT geometry)"; python -u tools/extract_unimol_pair.py \
      --sdf-path unimol_out_qm9mu_dft/sdf/qm9mu_for_unimol.sdf \
      --output-dir unimol_out_qm9mu_dft/pair_rep --ckpt-path "$CKPT" --dict-path "$DICT" --batch-size 8
  echo "[step3] make_qm9mu (DFT dirs)"; python -u make_qm9mu_from_unimol.py
fi
[ -f "$DATA" ] || { echo "DATA BUILD FAILED -- aborting"; exit 1; }
echo "[data ready] $DATA"
[ "${PREP_ONLY:-0}" = "1" ] && { echo "PREP_ONLY complete -- DFT data built, exiting before pilot"; exit 0; }

C="--task qm9mu --path down_task_qm9mu_unimol_v2_dft --finetune --epochs 1000 --eval_seeds 9,19,29 --batch_size 1024 --K 10 --hid_dim 512 --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3"
declare -a NAMES=(V0 V2T5 random T7 T7random T8 T8random)
declare -a FLAGS=("" "--use_v2" "--use_v2 --randomize_pair" "--use_v2 --t7" "--use_v2 --t7 --randomize_pair" "--use_v2 --t8" "--use_v2 --t8 --randomize_pair")
L=hpc/logs; N=${#NAMES[@]}
run_one(){ local nm=$1 fl=$2; echo "  [launch] $nm ($(date +%H:%M:%S))"
  python -u main_pretrain.py $C $fl > $L/qm9mu_dft_${nm}.log 2>&1; echo "JOBDONE $nm rc=$?" >> $L/qm9mu_dft_${nm}.log; }

if [ -n "${ARM:-}" ]; then
  echo "[step4] single arm: $ARM"
  hit=0
  for idx in $(seq 0 $((N-1))); do
    [ "${NAMES[$idx]}" = "$ARM" ] && { run_one "$ARM" "${FLAGS[$idx]}"; hit=1; }
  done
  [ $hit -eq 1 ] || { echo "unknown ARM=$ARM"; exit 2; }
else
  echo "[step4] all 7 arms (4 concurrent; may exceed 8h wall -- prefer ARM= split)"
  MAXC=4; running=0
  for idx in $(seq 0 $((N-1))); do
    ( run_one "${NAMES[$idx]}" "${FLAGS[$idx]}" ) &
    running=$((running+1)); [ $running -ge $MAXC ] && { wait -n 2>/dev/null; running=$((running-1)); }
  done
  wait
fi

echo "=== QM9-mu DFT test RMSE (finetune, lower=better) ==="
for nm in V0 V2T5 random T7 T7random T8 T8random; do printf "%-10s " "$nm:"; grep -hE "Best Test RMSE" $L/qm9mu_dft_${nm}.log | sed -E 's/.*is ([0-9.]+) .*/\1/' | tr '\n' ' '; echo; done
echo "QM9MU_DFT_HPC_DONE $(date)"
