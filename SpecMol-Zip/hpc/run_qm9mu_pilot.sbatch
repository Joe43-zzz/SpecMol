#!/usr/bin/env bash
#SBATCH --job-name=qm9mu
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# QM9 dipole(mu) 7-arm geometry-control pilot on HPC.
# Chain: prepare (download qm9.csv + RDKit embed 15k) -> extract Uni-Mol pair
#        -> make down_task_qm9mu_unimol_v2 -> 7 arms (max 4 concurrent on the 1 GPU,
#        ~3-4GB each, 32GB fits). DOES NOT set CUDA_VISIBLE_DEVICES (Slurm owns it).
# Question: does the geometry effect (real-pair vs shuffled-pair) AMPLIFY on the
# geometry-pure dipole target vs QM7 atomization energy (T7 geom~0, T8 geom~3.5)?
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

DATA=down_task_qm9mu_unimol_v2/processed/qm9mu_all.pt
if [ ! -f "$DATA" ]; then
  echo "[step1] prepare_qm9mu (download + embed 15k)"; python -u prepare_qm9mu_unimol_input.py
  echo "[step2] extract Uni-Mol pair"; python -u tools/extract_unimol_pair.py \
      --sdf-path unimol_out_qm9mu/sdf/qm9mu_for_unimol.sdf \
      --output-dir unimol_out_qm9mu/pair_rep --ckpt-path "$CKPT" --dict-path "$DICT" --batch-size 8
  echo "[step3] make_qm9mu"; python -u make_qm9mu_from_unimol.py
fi
[ -f "$DATA" ] || { echo "DATA BUILD FAILED -- aborting"; exit 1; }
echo "[data ready] $DATA"
[ "${PREP_ONLY:-0}" = "1" ] && { echo "PREP_ONLY complete -- data built, exiting before pilot"; exit 0; }

echo "[step4] 7-arm pilot (max 4 concurrent on 1 GPU)"
C="--task qm9mu --path down_task_qm9mu_unimol_v2 --finetune --epochs 1000 --eval_seeds 9,19,29 --batch_size 1024 --K 10 --hid_dim 512 --ft_max_epochs 40 --ft_patience 15 --ft_gate_lr 1e-2 --ft_encoder_lr 1e-4 --ft_head_lr 1e-3"
declare -a NAMES=(V0 V2T5 random T7 T7random T8 T8random)
declare -a FLAGS=("" "--use_v2" "--use_v2 --randomize_pair" "--use_v2 --t7" "--use_v2 --t7 --randomize_pair" "--use_v2 --t8" "--use_v2 --t8 --randomize_pair")
L=hpc/logs; MAXC=4
N=${#NAMES[@]}; running=0
for idx in $(seq 0 $((N-1))); do
  nm=${NAMES[$idx]}; fl=${FLAGS[$idx]}
  echo "  [launch] $nm  ($(date +%H:%M:%S))"
  ( python -u main_pretrain.py $C $fl > $L/qm9mu_${nm}.log 2>&1; echo "JOBDONE $nm rc=$?" >> $L/qm9mu_${nm}.log ) &
  running=$((running+1))
  if [ $running -ge $MAXC ]; then wait -n 2>/dev/null; running=$((running-1)); fi
done
wait

echo "=== QM9-mu RMSE (finetune, lower=better) ==="
for nm in V0 V2T5 random T7 T7random T8 T8random; do printf "%-10s " "$nm:"; grep -hE "Best Test RMSE" $L/qm9mu_${nm}.log | sed -E 's/.*is ([0-9.]+) .*/\1/' | tr '\n' ' '; echo; done
echo "QM9MU_HPC_DONE $(date)"
