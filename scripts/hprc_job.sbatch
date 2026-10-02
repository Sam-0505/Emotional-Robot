#!/usr/bin/env bash
#SBATCH --job-name=reachy-av
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -Eeuo pipefail

: "${SLURM_JOB_ID:?Submit this file with sbatch, not on a login node}"
: "${VLA_STAGE:?Missing stage}"
: "${VLA_REPO_ROOT:?Missing repository path}"
: "${VLA_DATASET:?Missing CREMA-D path}"
: "${VLA_RUN_ROOT:?Missing run path}"
: "${VLA_VENV:?Missing virtual environment path}"

cd "$VLA_REPO_ROOT"
if [[ -n "${VLA_MODULES:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
      if [[ -f "$module_init" ]]; then source "$module_init"; break; fi
    done
  fi
  type module >/dev/null 2>&1 || { echo "Grace module command is unavailable" >&2; exit 2; }
  read -r -a requested_modules <<< "$VLA_MODULES"
  module load "${requested_modules[@]}"
fi
source "$VLA_VENV/bin/activate"
export PYTHONPATH="$VLA_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$VLA_RUN_ROOT/hf_cache"
mkdir -p "$HF_HOME"

manifest="$VLA_RUN_ROOT/prepared/manifest.jsonl"
case "$VLA_STAGE" in
  unified-pilot|unified-full|unified-evaluate|unified-test)
    test -f "$manifest" || { echo "Prepared manifest missing: $manifest" >&2; exit 2; }
    python scripts/check_hprc.py "$VLA_DATASET" --training \
      --visual-revision "$VLA_VISUAL_REVISION" --audio-revision "$VLA_AUDIO_REVISION"
    joint_root="$VLA_RUN_ROOT/unified"
    if [[ "$VLA_STAGE" == unified-pilot || "$VLA_STAGE" == unified-full ]]; then
      joint_args=(--manifest "$manifest" --visual-revision "$VLA_VISUAL_REVISION" \
        --audio-revision "$VLA_AUDIO_REVISION" --config "$VLA_UNIFIED_CONFIG" --seed "$VLA_SEED")
      if [[ "$VLA_STAGE" == unified-pilot ]]; then
        joint_args+=(--output "$joint_root/pilot" --max-steps 1)
      else
        joint_args+=(--output "$joint_root/full" --max-steps 0 --epochs "$VLA_UNIFIED_EPOCHS" \
          --gradient-accumulation "$VLA_GRAD_ACCUM" --augment --pilot "$joint_root/pilot")
      fi
      python -u -m scripts.train_unified "${joint_args[@]}"
    else
      joint_split=validation
      calibration_args=()
      if [[ "$VLA_STAGE" == unified-test ]]; then
        joint_split=test
        calibration_args+=(--calibration "$joint_root/validation/unified_calibration.json")
      fi
      python -u -m scripts.evaluate_unified --manifest "$manifest" --checkpoint "$joint_root/full" \
        --split "$joint_split" --output "$joint_root/$joint_split" --ablations "${calibration_args[@]}"
    fi
    ;;
  prepare)
    if [[ -e "$VLA_RUN_ROOT/prepared" ]]; then
      echo "Prepared output already exists; use a fresh run root." >&2
      exit 2
    fi
    python scripts/check_hprc.py "$VLA_DATASET"
    python scripts/prepare_cremad.py "$VLA_DATASET" "$VLA_RUN_ROOT/prepared" --dry-run --seed "$VLA_SEED"
    python scripts/prepare_cremad.py "$VLA_DATASET" "$VLA_RUN_ROOT/prepared" --seed "$VLA_SEED"
    python scripts/prepare_cremad.py "$VLA_DATASET" "$VLA_RUN_ROOT/prepared" --validate
    ;;
  pilot|full|evaluate)
    test -f "$manifest" || { echo "Prepared manifest missing: $manifest" >&2; exit 2; }
    python scripts/check_hprc.py "$VLA_DATASET" --training \
      --visual-revision "$VLA_VISUAL_REVISION" --audio-revision "$VLA_AUDIO_REVISION"
    if [[ "$VLA_STAGE" == pilot ]]; then
      if [[ -e "$VLA_RUN_ROOT/pilot" ]]; then
        echo "Pilot output already exists; use a fresh run root." >&2
        exit 2
      fi
      mkdir "$VLA_RUN_ROOT/pilot"
      python -m scripts.train_visual --manifest "$manifest" --output "$VLA_RUN_ROOT/pilot/visual_adapter" \
        --revision "$VLA_VISUAL_REVISION" --max-steps 1 --seed "$VLA_SEED"
      python -m scripts.train_audio --manifest "$manifest" --output "$VLA_RUN_ROOT/pilot/audio_head" \
        --revision "$VLA_AUDIO_REVISION" --max-steps 1 --seed "$VLA_SEED"
      sample_id=$(python scripts/select_prepared_clip.py --manifest "$manifest" --split validation)
      python scripts/run_pipeline.py "$manifest" "$sample_id" \
        --visual-revision "$VLA_VISUAL_REVISION" --visual-adapter "$VLA_RUN_ROOT/pilot/visual_adapter" \
        --audio-revision "$VLA_AUDIO_REVISION" --audio-head "$VLA_RUN_ROOT/pilot/audio_head" \
        --output "$VLA_RUN_ROOT/pilot/observation.json"
      python scripts/verify_hprc_pilot.py --run-root "$VLA_RUN_ROOT" \
        --visual-revision "$VLA_VISUAL_REVISION" --audio-revision "$VLA_AUDIO_REVISION"
    elif [[ "$VLA_STAGE" == full ]]; then
      python scripts/verify_hprc_pilot.py --run-root "$VLA_RUN_ROOT" \
        --visual-revision "$VLA_VISUAL_REVISION" --audio-revision "$VLA_AUDIO_REVISION"
      if [[ -e "$VLA_RUN_ROOT/full" ]]; then
        echo "Full-training output already exists; use a fresh run root." >&2
        exit 2
      fi
      mkdir "$VLA_RUN_ROOT/full"
      python -m scripts.train_visual --manifest "$manifest" --output "$VLA_RUN_ROOT/full/visual_adapter" \
        --revision "$VLA_VISUAL_REVISION" --max-steps 0 --epochs "$VLA_VISUAL_EPOCHS" \
        --gradient-accumulation "$VLA_GRAD_ACCUM" --seed "$VLA_SEED" --augment
      python -m scripts.train_audio --manifest "$manifest" --output "$VLA_RUN_ROOT/full/audio_head" \
        --revision "$VLA_AUDIO_REVISION" --epochs "$VLA_AUDIO_EPOCHS" --seed "$VLA_SEED" --augment
    else
      python scripts/verify_hprc_pilot.py --run-root "$VLA_RUN_ROOT" --stage full \
        --visual-revision "$VLA_VISUAL_REVISION" --audio-revision "$VLA_AUDIO_REVISION"
      if [[ -e "$VLA_RUN_ROOT/evaluation" ]]; then
        echo "Evaluation output already exists; use a fresh run root." >&2
        exit 2
      fi
      zero_shot_args=()
      if [[ "${VLA_ZERO_SHOT:-0}" == 1 ]]; then zero_shot_args+=(--zero-shot-visual); fi
      python -m scripts.evaluate_perception --manifest "$manifest" \
        --visual-revision "$VLA_VISUAL_REVISION" --visual-adapter "$VLA_RUN_ROOT/full/visual_adapter" \
        --audio-revision "$VLA_AUDIO_REVISION" --audio-head "$VLA_RUN_ROOT/full/audio_head" \
        --output "$VLA_RUN_ROOT/evaluation" "${zero_shot_args[@]}"
    fi
    ;;
  *) echo "Unknown stage: $VLA_STAGE" >&2; exit 2 ;;
esac
