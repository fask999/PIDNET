#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash run_ablation.sh {show|train|eval} {cityscapes|camvid} PRESET [KEY VALUE ...]' \
    'Presets: baseline pappm dappm pmpm denseaspp valid-edge void-mask dice boundary fixed1 fixed2 fadc combo-fixed combo-fadc' \
    'show prints the training command. PIDNET_PYTHON selects Python; PIDNET_RUN_TAG distinguishes repeats.'
}

if (( $# < 3 )); then usage >&2; exit 2; fi
phase=$1
dataset=$2
preset=$3
shift 3
if (( $# % 2 != 0 )); then usage >&2; exit 2; fi

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
exp=test/20261111-pidnet-ablation-switches
case "$dataset" in
  cityscapes) cfg=$exp/configs/pidnet_medium_cityscapes.yaml ;;
  camvid) cfg=$exp/configs/pidnet_small_camvid.yaml ;;
  *) usage >&2; exit 2 ;;
esac

opts=()
boundary=(ABLATION.BOUNDARY.TARGET valid ABLATION.BOUNDARY.MASK_VOID True ABLATION.BOUNDARY.DICE_WEIGHT 1.0)
case "$preset" in
  baseline) ;;
  pappm|dappm|pmpm|denseaspp) opts=(ABLATION.CONTEXT "$preset") ;;
  valid-edge) opts=(ABLATION.BOUNDARY.TARGET valid) ;;
  void-mask) opts=(ABLATION.BOUNDARY.MASK_VOID True) ;;
  dice) opts=(ABLATION.BOUNDARY.DICE_WEIGHT 1.0) ;;
  boundary) opts=("${boundary[@]}") ;;
  fixed1) opts=(ABLATION.I_ATROUS.MODE fixed ABLATION.I_ATROUS.RATE 1) ;;
  fixed2) opts=(ABLATION.I_ATROUS.MODE fixed ABLATION.I_ATROUS.RATE 2) ;;
  fadc) opts=(ABLATION.I_ATROUS.MODE fadc) ;;
  combo-fixed) opts=(ABLATION.CONTEXT pmpm ABLATION.I_ATROUS.MODE fixed "${boundary[@]}") ;;
  combo-fadc) opts=(ABLATION.CONTEXT denseaspp ABLATION.I_ATROUS.MODE fadc "${boundary[@]}") ;;
  *) usage >&2; exit 2 ;;
esac

run=$preset
if [[ -n ${PIDNET_RUN_TAG:-} ]]; then
  if [[ ! $PIDNET_RUN_TAG =~ ^[a-z0-9-]+$ ]]; then
    printf '%s\n' 'PIDNET_RUN_TAG must contain only lowercase letters, numbers or hyphens.' >&2
    exit 2
  fi
  run=$run-$PIDNET_RUN_TAG
fi
out=$exp/output/$dataset/$run
logs=$exp/logs/$dataset/$run
py=${PIDNET_PYTHON:-python}
case "$phase" in
  show|train) entry=(tools/train.py --cfg "$cfg" --seed 304) ;;
  eval) entry=(tools/eval.py --cfg "$cfg") ;;
  *) usage >&2; exit 2 ;;
esac
cmd=("$py" -B "${entry[@]}" "${opts[@]}" "$@" OUTPUT_DIR "$out" LOG_DIR "$logs")

if [[ $phase == show ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  mkdir -p "$out" "$logs"
  log=$(mktemp "$logs/cli-$phase.XXXXXX.log")
  { printf '%q ' "${cmd[@]}"; printf '\n'; "${cmd[@]}"; } 2>&1 | tee "$log"
fi
