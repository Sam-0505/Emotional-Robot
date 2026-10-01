#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Usage: bash scripts/hprc_submit.sh --stage prepare|pilot|full|evaluate --dataset ABS_PATH --run-root ABS_PATH --venv ABS_PATH [options]" >&2
  echo "Options: --visual-revision SHA --audio-revision SHA --partition NAME --account NAME --gres SPEC --time HH:MM:SS --cpus N --mem SIZE --module NAME (repeatable) --setup-env --visual-epochs N --audio-epochs N --gradient-accumulation N --seed N --zero-shot --dry-run" >&2
}

stage= dataset= run_root= venv= visual_revision= audio_revision= account= partition= gres= time_limit= cpus=8 mem=64G
visual_epochs=1 audio_epochs=1 grad_accum=1 seed=42 zero_shot=0 dry_run=0 setup_env=0
modules=()
while (( $# )); do
  case "$1" in
    --stage|--dataset|--run-root|--venv|--visual-revision|--audio-revision|--account|--partition|--gres|--time|--cpus|--mem|--module|--visual-epochs|--audio-epochs|--gradient-accumulation|--seed)
      if (( $# < 2 )); then usage; exit 2; fi
      option=$1; value=$2; shift 2
      case "$option" in
        --stage) stage=$value ;; --dataset) dataset=$value ;; --run-root) run_root=$value ;;
        --venv) venv=$value ;; --visual-revision) visual_revision=$value ;;
        --audio-revision) audio_revision=$value ;; --account) account=$value ;;
        --partition) partition=$value ;; --gres) gres=$value ;; --time) time_limit=$value ;;
        --cpus) cpus=$value ;; --mem) mem=$value ;; --module) modules+=("$value") ;;
        --visual-epochs) visual_epochs=$value ;; --audio-epochs) audio_epochs=$value ;;
        --gradient-accumulation) grad_accum=$value ;; --seed) seed=$value ;;
      esac
      ;;
    --zero-shot) zero_shot=1; shift ;;
    --setup-env) setup_env=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

for path in "$dataset" "$run_root" "$venv"; do
  if [[ "$path" != /* || "$path" == / ]]; then
    echo "Dataset, run root and venv must be non-root absolute paths" >&2; exit 2
  fi
done
if [[ ! -d "$dataset" ]]; then echo "Dataset directory not found: $dataset" >&2; exit 2; fi
if (( ! setup_env )) && [[ ! -f "$venv/bin/activate" ]]; then
  echo "Python venv not found: $venv (use --setup-env to create and install it)" >&2; exit 2
fi
if [[ ! "$cpus" =~ ^[1-9][0-9]*$ || ! "$visual_epochs" =~ ^[1-9][0-9]*$ || ! "$audio_epochs" =~ ^[1-9][0-9]*$ || ! "$grad_accum" =~ ^[1-9][0-9]*$ || ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "CPU count, epochs, accumulation must be positive integers; seed must be nonnegative" >&2; exit 2
fi
if [[ "$stage" == prepare ]]; then
  if [[ -z "$partition" ]]; then echo "Choose a Grace CPU partition with --partition (check sinfo)" >&2; exit 2; fi
  if [[ -n "$gres" ]]; then echo "Preparation is a CPU stage; omit --gres" >&2; exit 2; fi
  time_limit=${time_limit:-12:00:00}
elif [[ "$stage" == pilot || "$stage" == full || "$stage" == evaluate ]]; then
  if [[ ! "$visual_revision" =~ ^[0-9a-f]{40}$ || ! "$audio_revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "GPU stages require both full 40-character lowercase checkpoint commit SHAs" >&2; exit 2
  fi
  partition=${partition:-gpu}
  gres=${gres:-gpu:a100:1}
  case "$stage" in pilot) time_limit=${time_limit:-12:00:00} ;; *) time_limit=${time_limit:-3-00:00:00} ;; esac
else
  echo "Unknown stage: $stage" >&2; usage; exit 2
fi
if [[ ! "$time_limit" =~ ^([0-9]+-)?[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$ || ! "$mem" =~ ^[1-9][0-9]*[GM]$ ]]; then
  echo "Use --time [D-]HH:MM:SS and --mem such as 64G" >&2; exit 2
fi
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
for protected_path in "$dataset" "$venv" "$repo_root"; do
  if [[ "$run_root/" == "$protected_path/"* ]]; then
    echo "Run root must not be within the dataset, venv or repository: $protected_path" >&2; exit 2
  fi
done
job_file="$repo_root/scripts/hprc_job.sbatch"
command=(sbatch --parsable --export=ALL --partition "$partition" --time "$time_limit" --cpus-per-task "$cpus" --mem "$mem")
if [[ -n "$account" ]]; then command+=(--account "$account"); fi
if [[ -n "$gres" ]]; then command+=(--gres "$gres"); fi
command+=(--output "$run_root/logs/reachy-%j.out" "$job_file")

echo "Stage: $stage; Grace partition: $partition; run root: $run_root"
printf 'Submit command: '; printf '%q ' "${command[@]}"; echo
if (( dry_run )); then
  if (( setup_env )); then echo "Would create/update $venv from requirements-hprc.txt"; fi
  echo "Dry run only: no directories created, dependencies installed, or job submitted"
  exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then echo "sbatch is unavailable; run this on a Grace login node" >&2; exit 2; fi
if (( ${#modules[@]} )); then
  for selected_module in "${modules[@]}"; do
    if [[ -z "$selected_module" || "$selected_module" == *[[:space:]]* ]]; then
      echo "Module names must be nonempty and contain no whitespace" >&2; exit 2
    fi
  done
  if ! type module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh; do
      if [[ -f "$module_init" ]]; then source "$module_init"; break; fi
    done
  fi
  type module >/dev/null 2>&1 || { echo "Grace module command is unavailable" >&2; exit 2; }
  module load "${modules[@]}"
  VLA_MODULES="${modules[*]}"; export VLA_MODULES
else
  VLA_MODULES=; export VLA_MODULES
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ffmpeg and ffprobe must both be available after module loading; pass the selected FFmpeg module with --module" >&2
  exit 2
fi
if (( setup_env )); then
  if [[ ! -f "$venv/bin/activate" ]]; then
    mkdir -p "$(dirname "$venv")"
    python -m venv "$venv"
  fi
  (cd "$repo_root" && "$venv/bin/python" -m pip install -r requirements-hprc.txt)
fi
if [[ ! -x "$venv/bin/python" ]]; then echo "Python is missing from venv: $venv" >&2; exit 2; fi
if ! "$venv/bin/python" -c 'import importlib.util, sys; required = ("cv2", "numpy", "torch", "transformers", "accelerate", "peft", "soundfile", "PIL", "timm", "einops", "open_clip"); missing = [name for name in required if importlib.util.find_spec(name) is None]; print("Missing Python dependencies: " + ", ".join(missing), file=sys.stderr) if missing else None; sys.exit(bool(missing))'; then
  echo "Install the Grace dependencies with --setup-env before submitting" >&2
  exit 2
fi
mkdir -p "$run_root/logs"
export VLA_STAGE="$stage" VLA_REPO_ROOT="$repo_root" VLA_DATASET="$dataset" VLA_RUN_ROOT="$run_root" VLA_VENV="$venv"
export VLA_VISUAL_REVISION="$visual_revision" VLA_AUDIO_REVISION="$audio_revision"
export VLA_VISUAL_EPOCHS="$visual_epochs" VLA_AUDIO_EPOCHS="$audio_epochs" VLA_GRAD_ACCUM="$grad_accum" VLA_SEED="$seed" VLA_ZERO_SHOT="$zero_shot"
"${command[@]}"
