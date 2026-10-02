# Reachy Emotions Prototype

An audio-visual acted-expression prototype for Reachy Mini. The unified U0 model pairs three face frames and matching speech from [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D). Frozen NVIDIA `Llama-3.1-Nemotron-Nano-VL-8B-V1` vision features and frozen WavLM speech features enter **one Nemotron decoder context**. A trainable audio projector and decoder LoRA learn jointly against `MultiModalVote`. Visual-only, audio-only, and late-fusion models remain comparison baselines. A small **agent harness** then asks a separate Nemotron reasoning model for a short response, validates its proposed text, Magpie speech style, and reviewed Reachy emotion move, and runs only approved effects.

There is no physical robot in this project environment. The intended motion target is the official Reachy Mini MuJoCo simulator through the high-level SDK. The user reported successful visual-baseline and unified one-step pilots on a Colab A100 40 GB, including the unified projector/LoRA gradient checks, modality ablations, and exact fresh-process score reload. Held-out accuracy, the Nebius call, Magpie deployment, and simulator run remain unverified. CREMA-D portrays *acted* expressions; this system does not determine someone's inner emotional state and is not a clinical tool.

## Architecture

```text
CREMA-D clip -> paired face frames + 16 kHz speech
             -> frozen Nemotron vision encoder + visual projector -> frame tokens
             -> frozen WavLM + trainable audio projector           -> speech tokens
             -> one Nemotron decoder with LoRA (MultiModalVote)
             -> validation calibration + paired-quality gates -> label or unknown
             -> agent harness: capability discovery -> Nemotron proposal
                              -> deterministic guard -> Magpie WAV + Reachy move
                              -> trace, or no action
```

The harness is application code, not a second model. It makes a single bounded decision; the reasoning model cannot execute arbitrary tools, access joint controls, or expand the move allowlist. A missing modality, abstention, invalid plan, stale input, unavailable voice/move, or operator stop prevents a robot response.

## Install and local smoke test

Use Python 3.10 or newer. Install FFmpeg (`ffmpeg` and `ffprobe`) for media preparation. Model training/inference needs a CUDA GPU with sufficient memory; `pip install -e '.[all]'` installs the optional Python dependencies. The Nemotron checkpoint uses custom Hugging Face model code, so pin and review its exact revision before loading it. The perception dependency group pins Transformers 4.57.3 and PEFT 0.18.0 because the checkpoint's custom loader is incompatible with the Transformers 5 loading path; re-install the group after updating an older environment.

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

## Optional baseline feasibility and evaluation

The standalone visual/audio classifiers and late fusion are retained for optional comparisons. They are not part of the joint inference path and are not prerequisites for joint training. Skip this section to focus on U0.

Choose and record full 40-character Hugging Face commit SHAs for both checkpoints; moving refs such as `main` are rejected. The visual path trains decoder attention LoRA against `FaceVote`; the frozen WavLM head uses `VoiceVote`. The visual script defaults to one optimizer step and verifies its saved adapter in a separate process. For the audio feasibility gate, pass `--max-steps 1` explicitly; its default is one full epoch. Inspect and archive the exact WavLM checkpoint license before downloading or training it.

```bash
python -m scripts.train_visual --manifest data/cremad/manifest.jsonl --output artifacts/visual_adapter --revision VISUAL_COMMIT --max-steps 1
python -m scripts.train_audio --manifest data/cremad/manifest.jsonl --output artifacts/audio_head --revision AUDIO_COMMIT --max-steps 1
python -m scripts.evaluate_perception --manifest data/cremad/manifest.jsonl --visual-revision VISUAL_COMMIT --visual-adapter artifacts/visual_adapter --audio-revision AUDIO_COMMIT --audio-head artifacts/audio_head --zero-shot-visual --output artifacts/evaluation
```

The optional baseline evaluation command predicts on validation/test actors, fits modality temperatures and the fusion weight on **validation only**, and reports held-out test metrics separately for `FaceVote`, `VoiceVote`, and `MultiModalVote`. `--zero-shot-visual` also scores the untouched NVIDIA checkpoint on the same clips; omit `--visual-adapter` to evaluate V0 by itself. It writes predictions and `fusion_config.json`. Do not treat a one-step adapter as a useful fine-tune. Baseline comparisons are needed to claim improvement over those baselines, not to train or report the joint model's held-out accuracy. For a longer visual run, use `--max-steps 0 --epochs N --gradient-accumulation K --seed 42 --augment` with a fresh adapter output directory. The optional visual augmentation varies framing, mirror, brightness, contrast, and color **only in training**; it is not a substitute for a background-shortcut audit.

The fusion weight is now selected by validation **macro-F1**, with abstentions counted as misses and both modalities assigned nonzero weight in deployable calibration. The evaluation command saves `validation_metrics.json` and `test_metrics.json` separately and rejects prediction files whose clip IDs do not exactly match the held-out manifest. For a longer audio-head run, add `--epochs N --seed 42 --augment`; augmentation is training-only mild gain, noise, and bandwidth variation. Codec augmentation remains unimplemented. The one-step feasibility runs should remain unaugmented for easier debugging.

To create one robot-facing observation from a paired clip:

```bash
python scripts/run_pipeline.py data/cremad/manifest.jsonl SAMPLE_ID --visual-revision VISUAL_COMMIT --visual-adapter artifacts/visual_adapter --audio-revision AUDIO_COMMIT --audio-head artifacts/audio_head --fusion-config artifacts/evaluation/fusion_config.json --output artifacts/observation.json
```

This verifies the prepared frame/audio hashes, loads both real experts, and marks a clip action-eligible only if fusion does not abstain, at least two of three frames contain a detected face, speech activity is present, and audio/video durations align. These are conservative prototype gates, not a validated live-quality classifier. Observations expire after ten seconds in the default runtime guard; a saved observation file is primarily for inspection. The integrated command below infers and responds in one process.

## Unified audio-visual model (U0)

The implementation extends the existing VLM with waveform features; stock Nemotron does not accept raw audio. It calls the language decoder with `inputs_embeds`, bypassing the stock image-only forward. Defaults in [configs/unified_u0.json](configs/unified_u0.json) use one image tile per frame, spatially pool each frame to 64 tokens, and temporally pool WavLM to 16 speech tokens before a trainable two-layer projector. The three frames remain ordered in the prompt. Frozen encoders run without gradients; only the audio projector and decoder attention LoRA train. Instruction, visual, and audio positions are excluded from the answer loss. No actor ID, filename, transcript, or crowd-vote label enters the inference prompt.

The visual pooling changes the original token layout intentionally to keep the joint sequence small. Native 256 tokens per frame can be selected in a separate configuration/run if validation supports the additional cost. Gradient checkpointing is enabled by default. No automatic late-fusion fallback occurs on an out-of-memory error.

The single-tile/no-thumbnail controls are set on the processor instance, not passed as unsupported call-time keywords. The wrapper checks that each frame produces one feature tile. After updating an older checkout that reported ignored tiling keywords, use `--stage verify` to recheck the saved pilot's scores without retraining.

The same prepared manifest is reused. Training selects only synchronized, usable training clips with an unambiguous `MultiModalVote`; FaceVote/VoiceVote do not supervise U0. Missing faces are masked, and fewer than two usable frames, poor speech, or bad synchronization prevents a joint prediction. The decoder scores six fixed JSON completions; JSON is rendered from the selected label, so this is constrained classification, not unconstrained JSON generation or speech transcription.

### Colab: joint pilot with existing MyDrive data

After pulling the repository, run these notebook cells:

```python
%cd /content/drive/MyDrive/reachy-av/Emotional-Robot
%pip install -e '.[perception]'
```

```python
%run scripts/run_unified_colab.py --stage pilot
```

The launcher uses the notebook's Python, reads `nemotron-revision.txt` and `wavlm-revision.txt`, reuses `cremad-prepared-run2/manifest.jsonl` and `huggingface-cache`, and saves everything under `MyDrive/reachy-av/unified-001`. It locates the repository's `src` directory for both notebook-side imports and training subprocesses; the perception dependencies must still be installed in the notebook's Python. Use `--base`, `--manifest`, or `--run-root` to override these paths. `--dry-run` prints the command without creating outputs. Logs stream into the notebook and a timestamped Drive file.

The pilot performs one real optimizer step, checks nonzero finite gradients in both trainable components and no gradients in frozen weights, records feature shapes, step time and peak allocated VRAM, and checks that masking either modality affects scores. It saves `decoder_lora/`, `audio_projector.pt`, and `unified_metadata.json`, then reloads both trainable components in a fresh process and reproduces joint and ablated scores. Ablations verify wiring only; they do not establish learned reliance or accuracy. If training saved successfully but verification was interrupted, use `--stage verify`. Partial runs are never overwritten; use a new `--run-root` if needed.

User-reported A100 pilot results: loss 1.1745, first-step time 54.26 seconds, peak allocated VRAM 17.19 GiB, gradient/ablation checks passed, and reload score difference 0.000000. These are feasibility results, not accuracy or steady-state throughput. Re-verify after the processor-settings correction above before full training.

The equivalent portable command is:

```bash
python -u -m scripts.train_unified --manifest PREPARED/manifest.jsonl --output RUN/pilot --visual-revision VISUAL_COMMIT --audio-revision AUDIO_COMMIT --config configs/unified_u0.json --max-steps 1
```

### Full joint training and held-out comparison

Full U0 training requires a verified joint pilot with matching revisions, manifest and architecture. It does **not** require standalone audio/visual training or baseline metrics. Train and then calibrate/evaluate the joint model on validation actors:

```bash
python -u -m scripts.train_unified --manifest PREPARED/manifest.jsonl --output RUN/full --visual-revision VISUAL_COMMIT --audio-revision AUDIO_COMMIT --config configs/unified_u0.json --max-steps 0 --epochs 1 --gradient-accumulation 4 --augment --pilot RUN/pilot
python -u -m scripts.evaluate_unified --manifest PREPARED/manifest.jsonl --checkpoint RUN/full --split validation --ablations --output RUN/validation
```

Without `--resume`, full training starts a fresh U0 experiment from the pinned base models. It does not silently reuse pilot/baseline adapters. Epoch count is explicit; validation/model selection occurs in the separate evaluation command. Automatic early stopping and frozen-feature caching are not implemented.

### Continue training or recover an interrupted joint run

The joint trainer saves resumable snapshots every 100 optimizer steps by default, at each epoch end, and at the final step. Change the interval with `--save-every N`. Snapshots contain decoder LoRA, the audio projector, AdamW state, Python/NumPy/Torch/CUDA and augmentation random states, exact next batch position, and accumulated loss/time history. Gradient accumulation is flushed at each epoch boundary. Completed snapshots live in `full/checkpoints/step-XXXXXXXX`; `full/last_checkpoint.json` points to the newest complete snapshot. A failed save leaves the previous pointer intact. Snapshots are retained, so budget Drive space; no older checkpoints are automatically deleted.

`--epochs` and `--max-steps` are **total targets**, not additional budgets. For example, to extend a completed two-epoch run to four total epochs in Colab:

```python
%run scripts/run_unified_colab.py --stage full --epochs 4 --resume /content/drive/MyDrive/reachy-av/unified-001/full --run-root /content/drive/MyDrive/reachy-av/unified-002
```

The launcher restores the saved seed, learning rate, accumulation, augmentation, and architecture. The trainer rechecks the original joint pilot, unless an explicit path overrides it. Optional baseline reports, including ones referenced by older checkpoints, are not required for resume. Use a **new output/run root**, including when recovering an interrupted run, to preserve prior weights, metrics and calibration. Point `--resume` at the previous training output (resolved through its pointer) or at an individual checkpoint folder. Recovery starts at the last saved optimizer boundary; an interrupted accumulation window and any unsaved steps are replayed.

The equivalent direct command must retain the original settings (example: accumulation 4, default learning rate/seed, augmentation enabled):

```bash
python -u -m scripts.train_unified --manifest PREPARED/manifest.jsonl --output CONTINUED/full --visual-revision VISUAL_COMMIT --audio-revision AUDIO_COMMIT --max-steps 0 --epochs 4 --gradient-accumulation 4 --augment --resume RUN/full
```

Resume rejects changed model revisions, architecture, manifest, seed, accumulation, learning rate, augmentation, PyTorch version, or CUDA device count. Exact CPU continuation is tested, including in a fresh process; CUDA kernel determinism is not guaranteed. New output weights must be recalibrated/evaluated on validation before test or harness use. The final inference artifacts still undergo fresh-process score verification. Earlier runs without `training_state.pt` cannot recover their missing optimizer state; this feature applies to runs made with the updated trainer. Baseline visual/audio trainers are unchanged.

Only after selecting a run using validation should you evaluate test:

```bash
python -u -m scripts.evaluate_unified --manifest PREPARED/manifest.jsonl --checkpoint RUN/full --split test --calibration RUN/validation/unified_calibration.json --ablations --output RUN/test
```

Omit `--baseline-predictions` for joint metrics alone. Comparison files must cover exactly the selected split's clips. Calibration uses only validation `MultiModalVote`, binds to artifact and manifest hashes, and is never refitted on test. Reports include macro-F1, accuracy, confusion, per-actor results, abstention and optional modality-ablated metrics. Recorded inference latency includes the ablation scoring passes when `--ablations` is enabled.

Colab equivalents are `--stage full --epochs 1`, `--stage validation`, and, after selection, `--stage test` with the launcher. Pass `--baseline-predictions PATH` only if you choose to make matched comparisons. `--baseline-evaluation PATH` is optional report provenance and is validated only when explicitly supplied. Do not present one-step pilot metrics as a trained model result.

### Joint perception into the existing harness

Both inference commands accept a joint checkpoint/calibration pair instead of the baseline model arguments:

```bash
python scripts/run_pipeline.py PREPARED/manifest.jsonl SAMPLE_ID --unified-checkpoint RUN/full --unified-calibration RUN/validation/unified_calibration.json --output RUN/observation.json
python scripts/run_integrated.py PREPARED/manifest.jsonl SAMPLE_ID --unified-checkpoint RUN/full --unified-calibration RUN/validation/unified_calibration.json
```

The second command runs perception only; the existing explicit `--execute` and service/allowlist options enable the guarded simulator response. The joint path retains prepared-media hash checks, calibrated abstention, observation freshness, and measured paired-quality gates. Uncalibrated joint outputs cannot authorize speech or motion.

## TAMU HPRC Grace batch run

The joint-model stages are `unified-pilot`, `unified-full`, `unified-evaluate` (validation), and `unified-test` (uses the saved validation calibration). They reuse the same run root's prepared manifest and checkpoint revisions; outputs go under `RUN_ROOT/unified`. Pass `--unified-config configs/unified_u0.json` and `--unified-epochs N` as needed. `unified-full` requires `RUN_ROOT/unified/pilot` to pass its checks; no baseline evaluation files are required for joint training or evaluation. The existing `pilot`, `full`, and `evaluate` stages remain optional baseline workflows.

```bash
bash scripts/hprc_submit.sh --stage unified-pilot --dataset "$DATASET" --run-root "$RUN_ROOT" --venv "$VENV" --visual-revision "$VISUAL_SHA" --audio-revision "$AUDIO_SHA" --module "$PYTHON_MODULE" --module "$FFMPEG_MODULE" --dry-run
```

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
