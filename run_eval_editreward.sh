#!/usr/bin/env bash
#SBATCH -J eval_editreward
#SBATCH -p gigabyte_a6000,suma_a6000
#SBATCH -q base_qos
#SBATCH --gres=gpu:1
#SBATCH --output=.logs/imgedit_editreward_%j.log
#SBATCH --time 24:00:00

# ``sbatch`` 는 보통 이 스크립트와 같은 디렉터리( EditReward )에서 실행. Slurm은 스크립트를 spool에 복사하므로
# BASH_SOURCE 로는 저장소 경로를 알 수 없음 → PWD / SLURM_SUBMIT_DIR 우선.
set -euo pipefail

SUB="${SLURM_SUBMIT_DIR:-}"
SUB="${SUB%/}"
if [[ -n "$SUB" && -f "$SUB/eval_editreward.py" ]]; then
    _ER="$SUB"
elif [[ -n "$SUB" && -f "$SUB/EditReward/eval_editreward.py" ]]; then
    _ER="$SUB/EditReward"
elif [[ -f "$PWD/eval_editreward.py" ]]; then
    _ER="$PWD"
else
    _ER="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
fi
REPO="$(dirname "$_ER")"

cd "$_ER" || exit 1

if [[ -f /scratch2/khmin1104/venvs/edit_reward/bin/activate ]]; then
    source /scratch2/khmin1104/venvs/edit_reward/bin/activate
else
    echo "WARNING: venv not found; venv edit_reward not activated." >&2
fi

exec python eval_editreward.py \
    --dataset magicbrush \
    --model flux_kontext \
    --sample-id-start 0 \
    --sample-id-end 582000 \
    "$@"
