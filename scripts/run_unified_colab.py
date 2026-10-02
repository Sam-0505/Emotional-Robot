"""Launch joint training/evaluation with streamed logs and durable MyDrive paths."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys


def build_command(args, repo):
    base = args.base.resolve()
    manifest = args.manifest or base / "cremad-prepared-run2/manifest.jsonl"
    run = args.run_root or base / "unified-001"
    visual_revision = (base / "nemotron-revision.txt").read_text().strip()
    audio_revision = (base / "wavlm-revision.txt").read_text().strip()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    common = [sys.executable, "-u", "-m"]
    if args.stage in ("pilot", "full", "verify"):
        checkpoint = run / ("full" if args.stage == "full" else "pilot")
        command = common + ["scripts.train_unified", "--manifest", str(manifest),
                            "--output", str(checkpoint), "--visual-revision", visual_revision,
                            "--audio-revision", audio_revision, "--config", str(args.config or repo / "configs/unified_u0.json")]
        if args.stage == "full":
            if not args.baseline_evaluation:
                raise ValueError("full requires --baseline-evaluation from the visual/audio/fusion validation run")
            command += ["--max-steps", "0", "--epochs", str(args.epochs), "--gradient-accumulation",
                        str(args.gradient_accumulation), "--augment", "--pilot", str(run / "pilot"),
                        "--baseline-evaluation", str(args.baseline_evaluation)]
        elif args.stage == "verify":
            command += ["--verify-only"]
        else:
            command += ["--max-steps", "1"]
    else:
        command = common + ["scripts.evaluate_unified", "--manifest", str(manifest),
                            "--checkpoint", str(run / "full"), "--split", args.stage,
                            "--output", str(run / args.stage), "--ablations"]
        if args.stage == "test":
            command += ["--calibration", str(run / "validation/unified_calibration.json")]
        if args.baseline_predictions:
            command += ["--baseline-predictions", str(args.baseline_predictions)]
    return command, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/content/drive/MyDrive/reachy-av"))
    parser.add_argument("--stage", choices=("pilot", "verify", "full", "validation", "test"), default="pilot")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--baseline-evaluation", type=Path)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    command, run = build_command(args, repo)
    print("Command:", shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["HF_HOME"] = str(args.base / "huggingface-cache")
    env["PYTHONUNBUFFERED"] = "1"
    # The experiment is entirely PyTorch; avoid unrelated Colab TF initialization.
    env["USE_TF"] = "0"
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = logs / (args.stage + "-" + stamp + ".log")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    print("Exit code:", code, "| Log:", log_path, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
