# Reachy Emotions Prototype

An audio-visual acted-expression prototype for Reachy Mini. It pairs three face frames and the matching speech waveform from [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D), fine-tunes NVIDIA's `Llama-3.1-Nemotron-Nano-VL-8B-V1` for the visual side, trains a WavLM speech classifier, and fuses their outputs. A small **agent harness** then asks a Nemotron reasoning model for a short response, validates its proposed text, Magpie speech style, and reviewed Reachy emotion move, and runs only approved effects.

There is no physical robot in this project environment. The intended motion target is the official Reachy Mini MuJoCo simulator through the high-level SDK. The current repository includes an offline harness demo and live adapters, but no actual GPU training, Nebius call, Magpie deployment, or simulator run has been verified here. CREMA-D portrays *acted* expressions; this system does not determine someone's inner emotional state and is not a clinical tool.

## Architecture

```text
CREMA-D clip -> paired face frames + 16 kHz speech
             -> NVIDIA Nemotron VL visual expert (FaceVote)
             -> WavLM audio expert (VoiceVote)
             -> quality-aware fusion (MultiModalVote or unknown)
             -> agent harness: capability discovery -> Nemotron proposal
                              -> deterministic guard -> Magpie WAV + Reachy move
                              -> trace, or no action
```

The harness is application code, not a second model. It makes a single bounded decision; the reasoning model cannot execute arbitrary tools, access joint controls, or expand the move allowlist. A missing modality, abstention, invalid plan, stale input, unavailable voice/move, or operator stop prevents a robot response.

## Install and local smoke test

Use Python 3.10 or newer. Install FFmpeg (`ffmpeg` and `ffprobe`) for media preparation. Model training/inference needs a CUDA GPU with sufficient memory; `pip install -e '.[all]'` installs the optional Python dependencies. The Nemotron checkpoint uses custom Hugging Face model code, so pin and review its exact revision before loading it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[all]'
python -m unittest discover -s tests -v
python scripts/run_demo.py
```

`run_demo.py` with no flags uses a mock observation, planner, silent WAV, and mock robot. It proves the harness control flow, **not** model quality or simulator behavior. It prints the decision trace and robot events.

## Prepare only CREMA-D

Obtain the dataset from the [official CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D) with Git LFS; keep the raw checkout outside Git tracking (for example, the ignored `data/CREMA-D/` directory). Follow its ODbL/DbCL attribution and license terms. The preparer reads the official `processedResults/summaryTable.csv`, `tabulatedVotes.csv`, and `VideoDemographics.csv`, checks for Git LFS pointers, and creates actor-disjoint splits with synchronized frames and mono PCM16 audio. Prepared audio receives one clip-wide gain toward 0.1 RMS measured from higher-energy 20 ms blocks, capped at 4× gain and a 0.95 peak. This leaves fillers and pauses in place, so a long gap does not by itself increase the gain. The manifest records quality before and after normalization, and review status uses the original high-energy level, activity fraction, and clipping so gain cannot make a poor source appear clean. This energy-based activity estimate is not voice activity detection or background-speaker removal. The preparer refuses to overwrite an existing manifest; choose a new output directory for a new preparation run.

```bash
python scripts/prepare_cremad.py /path/to/CREMA-D data/cremad --dry-run
python scripts/prepare_cremad.py /path/to/CREMA-D data/cremad
python scripts/prepare_cremad.py /path/to/CREMA-D data/cremad --validate
```

Preparation logs the first clip, every 100 clips by default, and the final clip with elapsed time and estimated time remaining. It then announces the separate manifest/media validation phase. Use `--log-every 10` for more frequent progress in Colab. Logs appear while the command runs. A failed preparation does not create a valid manifest or resume from partial output.

For the single verified local test pair, run `python scripts/smoke_media.py`. It checks the video and WAV bytes against the official Git LFS pointers, their media headers, and the CREMA-D vote tables without needing FFmpeg. On a host with FFmpeg and OpenCV, `python scripts/smoke_media.py --decode` also exercises the production three-frame/audio extraction in a temporary directory. A dry run of that exact clip is:

```bash
python scripts/prepare_cremad.py data/CREMA-D data/cremad_sample --clip-id 1001_DFA_ANG_XX --video-dir data --audio-dir data --dry-run
```

`--video-dir` and `--audio-dir` are only for explicitly selected clips and verify each override against the official repository media hash. Preparation uses official `VideoFlash` frames and the matching official `AudioWAV` waveform, with a duration-alignment check. This clip is in the **train** actor split under the current demographic-balanced seed, but a single clip remains only a decoding/one-step smoke test, not an evaluation dataset. On HPRC, use a full Git LFS checkout and the normal preparation command without media overrides; fetch `VideoFlash/*` and `AudioWAV/*`, then check storage headroom for the LFS cache and prepared output. The local clone still contains media pointers, not the full dataset.

On the HPRC worker, run `python scripts/check_hprc.py /path/to/CREMA-D` before preparation. After installing the model dependencies and choosing full 40-character checkpoint commits, run `python scripts/check_hprc.py /path/to/CREMA-D --training --visual-revision VISUAL_COMMIT --audio-revision AUDIO_COMMIT`. The check is read-only and reports missing tools, unresolved LFS pointers, dependencies, and CUDA visibility; it does not prove that the model fits in GPU memory.

The default split is deterministic and approximately 64/13/14 actors for train/validation/test. It is actor-disjoint and selects from seeded candidate splits to balance the official sex, race, ethnicity, and age-band marginals; it does not guarantee equal subgroup sizes or generalization. `split_balance.json` exposes label counts, `demographic_balance.json` exposes aggregate actor counts, `dataset_attribution.json` records the official source/license hashes, and `provenance.json` records metadata hashes, split strategy/seed, and preprocessing tool versions. OpenCV is needed for face cropping; clips with no detected face are marked in the manifest, and the robot-facing pipeline refuses to treat them as valid visual evidence. Do not commit raw or prepared media.

## One-step feasibility and evaluation

Choose and record full 40-character Hugging Face commit SHAs for both checkpoints; moving refs such as `main` are rejected. The visual path trains decoder attention LoRA against `FaceVote`; the frozen WavLM head uses `VoiceVote`. The visual script defaults to one optimizer step and verifies its saved adapter in a separate process. For the audio feasibility gate, pass `--max-steps 1` explicitly; its default is one full epoch. Inspect and archive the exact WavLM checkpoint license before downloading or training it.

```bash
python -m scripts.train_visual --manifest data/cremad/manifest.jsonl --output artifacts/visual_adapter --revision VISUAL_COMMIT --max-steps 1
python -m scripts.train_audio --manifest data/cremad/manifest.jsonl --output artifacts/audio_head --revision AUDIO_COMMIT --max-steps 1
python -m scripts.evaluate_perception --manifest data/cremad/manifest.jsonl --visual-revision VISUAL_COMMIT --visual-adapter artifacts/visual_adapter --audio-revision AUDIO_COMMIT --audio-head artifacts/audio_head --zero-shot-visual --output artifacts/evaluation
```

The evaluation command predicts on validation/test actors, fits modality temperatures and the fusion weight on **validation only**, and reports held-out test metrics separately for `FaceVote`, `VoiceVote`, and `MultiModalVote`. `--zero-shot-visual` also scores the untouched NVIDIA checkpoint on the same clips; omit `--visual-adapter` to evaluate V0 by itself. It writes predictions and `fusion_config.json`. Do not treat a one-step adapter as a useful fine-tune; run and compare the full predeclared baselines in [PROJECT.md](PROJECT.md) before making performance claims. For a longer visual run, use `--max-steps 0 --epochs N --gradient-accumulation K --seed 42 --augment` with a fresh adapter output directory. The optional visual augmentation varies framing, mirror, brightness, contrast, and color **only in training**; it is not a substitute for a background-shortcut audit.

The fusion weight is now selected by validation **macro-F1**, with abstentions counted as misses and both modalities assigned nonzero weight in deployable calibration. The evaluation command saves `validation_metrics.json` and `test_metrics.json` separately and rejects prediction files whose clip IDs do not exactly match the held-out manifest. For a longer audio-head run, add `--epochs N --seed 42 --augment`; augmentation is training-only mild gain, noise, and bandwidth variation. Codec augmentation remains unimplemented. The one-step feasibility runs should remain unaugmented for easier debugging.

To create one robot-facing observation from a paired clip:

```bash
python scripts/run_pipeline.py data/cremad/manifest.jsonl SAMPLE_ID --visual-revision VISUAL_COMMIT --visual-adapter artifacts/visual_adapter --audio-revision AUDIO_COMMIT --audio-head artifacts/audio_head --fusion-config artifacts/evaluation/fusion_config.json --output artifacts/observation.json
```

This verifies the prepared frame/audio hashes, loads both real experts, and marks a clip action-eligible only if fusion does not abstain, at least two of three frames contain a detected face, speech activity is present, and audio/video durations align. These are conservative prototype gates, not a validated live-quality classifier. Observations expire after ten seconds in the default runtime guard; a saved observation file is primarily for inspection. The integrated command below infers and responds in one process.

## TAMU HPRC Grace batch run

The [Grace batch system](https://hprc.tamu.edu/kb/User-Guides/Grace/Batch/) uses Slurm. The new [submission wrapper](scripts/hprc_submit.sh) submits separate `prepare` (CPU), `pilot` (one optimizer step per model plus paired inference), `full` (longer fine-tuning), and `evaluate` jobs through [the batch job](scripts/hprc_job.sbatch). Run the wrapper **from a Grace login node**, not the training command directly there. It does not submit later stages automatically. GPU stages default to Grace's documented `gpu` partition with one A100 (`gpu:a100:1`); preparation requires you to choose an available CPU partition with `--partition`. A100 availability and memory sufficiency are not guaranteed; inspect the pilot log before committing to a full run. The job's default CPU/memory/time requests are estimates and can be overridden on submission.

Put the complete Git LFS CREMA-D checkout, repository clone, and run output on Grace-accessible storage. [Scratch is not backed up](https://hprc.tamu.edu/kb/User-Guides/Grace/Filesystems_and_Files/), so copy final adapters, metrics, provenance, and logs to durable storage after the run. Python dependencies for these jobs are in [requirements-hprc.txt](requirements-hprc.txt), which uses the dependency groups in `pyproject.toml`. FFmpeg (`ffmpeg` and `ffprobe`) and Git LFS are **system executables**, so a pip requirements file cannot supply them reliably. Discover suitable Grace modules with `module spider`; the submission wrapper loads the selected Python and FFmpeg modules, checks both FFmpeg commands, and checks the venv dependencies before submitting. Its `--setup-env` option creates the venv if needed and installs `requirements-hprc.txt` once on the login node; omit it on later submissions unless you intentionally want to update the environment. Check the exact model licenses before downloading weights. The [HPRC Python guide](https://hprc.tamu.edu/kb/Software/Python/) covers module discovery and scratch-based venvs. Worker nodes lack normal internet access; [HPRC WebProxy](https://hprc.tamu.edu/kb/Software/WebProxy/) is needed if either pinned checkpoint is not already cached.

From the repository directory in the Grace portal terminal, discover module names that actually exist on your account (including any hierarchical prerequisites shown by `module spider`):

```bash
module spider Python
module spider FFmpeg
```

If Grace does not offer an FFmpeg module, arrange a user-space installation that puts **both** executables on `PATH` for the worker; do not substitute a Python package called `ffmpeg`. If PyTorch cannot see CUDA in the pilot job, resolve the wheel/CUDA compatibility before full training.

For example, set the paths and exact 40-character Hugging Face commit SHAs, then dry-run each command before dropping `--dry-run`:

```bash
DATASET="$SCRATCH/CREMA-D"
RUN_ROOT="$SCRATCH/reachy-av/run-001"
VENV="$SCRATCH/reachy-av/.venv"
VISUAL_SHA=REPLACE_WITH_40_CHARACTER_NVIDIA_COMMIT
AUDIO_SHA=REPLACE_WITH_40_CHARACTER_WAVLM_COMMIT
CPU_PARTITION=REPLACE_WITH_GRACE_CPU_PARTITION

PYTHON_MODULE=REPLACE_WITH_SELECTED_PYTHON_MODULE
FFMPEG_MODULE=REPLACE_WITH_SELECTED_FFMPEG_MODULE

bash scripts/hprc_submit.sh --stage prepare --dataset "$DATASET" --run-root "$RUN_ROOT" --venv "$VENV" --partition "$CPU_PARTITION" --module "$PYTHON_MODULE" --module "$FFMPEG_MODULE" --setup-env --dry-run
bash scripts/hprc_submit.sh --stage pilot --dataset "$DATASET" --run-root "$RUN_ROOT" --venv "$VENV" --visual-revision "$VISUAL_SHA" --audio-revision "$AUDIO_SHA" --module "$PYTHON_MODULE" --module "$FFMPEG_MODULE" --module WebProxy --dry-run
bash scripts/hprc_submit.sh --stage full --dataset "$DATASET" --run-root "$RUN_ROOT" --venv "$VENV" --visual-revision "$VISUAL_SHA" --audio-revision "$AUDIO_SHA" --visual-epochs 2 --audio-epochs 2 --gradient-accumulation 4 --module "$PYTHON_MODULE" --module "$FFMPEG_MODULE" --module WebProxy --dry-run
bash scripts/hprc_submit.sh --stage evaluate --dataset "$DATASET" --run-root "$RUN_ROOT" --venv "$VENV" --visual-revision "$VISUAL_SHA" --audio-revision "$AUDIO_SHA" --zero-shot --module "$PYTHON_MODULE" --module "$FFMPEG_MODULE" --module WebProxy --dry-run
```

Submit them **one at a time**, only after the previous job succeeds. `prepare` validates and extracts all official pairs; `pilot` checks saved adapters in separate processes and writes `pilot/observation.json`; `full` refuses to start unless the pilot used both modalities and the saved model revisions/manifest still match; `evaluate` requires verified full checkpoints and writes held-out metrics. Each stage refuses to overwrite its output, so use a new run root for a new experiment. The wrapper prints the `sbatch` job ID; inspect it with `squeue -j JOB_ID` and then read `$RUN_ROOT/logs/reachy-JOB_ID.out` or use `seff JOB_ID` after completion. Supply `--account ACCOUNT` if your allocation requires it. The submitted repository path must be visible from compute nodes, and the wrapper does not transfer or download CREMA-D for you.

## Live simulator adapters

Run the [official Reachy Mini MuJoCo daemon](https://github.com/pollen-robotics/reachy_mini/blob/main/docs/source/platforms/simulation/get_started.md), make the [official emotions library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) available in the SDK cache, and provide a running NVIDIA Magpie Speech NIM HTTP endpoint. Set `NEBIUS_API_KEY` and select an **actually available** Nemotron text model ID in Nebius Token Factory. Token Factory is for the reasoning planner; the visual adapter can be trained on Grace. The [competition rules](https://nebiusglobalaihackathon.devpost.com/rules) require a real Token Factory runtime call **or** Nebius AI Cloud deployment; HPRC training by itself is not sufficient. No model ID, voice, or motion name is assumed to exist without discovery.

```bash
python scripts/run_demo.py --live --nebius-model MODEL_ID --tts-url http://localhost:9000 --allow-move REVIEWED_MOVE --observation-json artifacts/observation.json
```

The preferred fresh end-to-end path is:

```bash
python scripts/run_integrated.py data/cremad/manifest.jsonl SAMPLE_ID --visual-revision VISUAL_COMMIT --visual-adapter artifacts/visual_adapter --audio-revision AUDIO_COMMIT --audio-head artifacts/audio_head --fusion-config artifacts/evaluation/fusion_config.json --execute --nebius-model MODEL_ID --tts-url http://localhost:9000 --allow-move REVIEWED_MOVE
```

Omit `--execute` to run perception without speech or motion. These live commands are integration hooks, not validated demos in this environment. Review each move before allowlisting it. The harness discovers currently available moves and voices, rejects unsupported plans before synthesis, and suppresses the emotion move's recorded audio sidecar in favor of generated speech. It logs a compact trace and client-side audio/motion **dispatch** offset; this is not a measurement of audible or physical onset. A prerecorded observation will be rejected if stale.

For a context-aware reply, add `--transcript 'Hello, Reachy'` and optionally `--recent-context 'We just started talking'` to either live command. These one-line inputs are bounded and sent to the Nebius reasoning model; do not supply private conversation text without consent. They are supplied text, **not** live speech recognition.

## Limits and next evidence

CREMA-D alone cannot establish generalization to natural interactions, new rooms, camera angles, microphones, or identities. The current actor split prevents direct identity overlap, but background and identity shortcut audits, training-only augmentations, calibration curves, and live input quality tests are still required. The repository does not include a synchronized live camera/microphone capture path, ROS 2, or Isaac Sim. MuJoCo is the primary simulator; ROS 2 and Isaac Sim remain optional after the core path is proven. Do not present offline mocks as hardware or cloud evidence.

See [PROJECT.md](PROJECT.md) for the experiment and implementation plan and [HACKATHON_REQUIREMENTS.md](HACKATHON_REQUIREMENTS.md) for submission requirements.

## Licenses and attribution

Project source is MIT-licensed in [LICENSE](LICENSE). CREMA-D is a separate dataset under its own ODbL/DbCL terms. The NVIDIA, Microsoft, Magpie, and Reachy checkpoints, services, datasets, and SDKs retain their own licenses and usage conditions; this repository does not redistribute their weights or media.
