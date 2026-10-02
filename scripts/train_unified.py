"""Train or verify one Nemotron decoder conditioned on paired video and audio."""

import argparse
import gc
import subprocess
import sys

from reachy_emotions.perception.unified import train_unified, verify_unified
from reachy_emotions.perception.unified_config import UnifiedConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visual-revision", required=True)
    parser.add_argument("--audio-revision", required=True)
    parser.add_argument("--config", help="Joint architecture JSON; defaults to U0 token budgets")
    parser.add_argument("--max-steps", type=int, default=1, help="Total optimizer-step cap including resumed steps; 0 means all epochs")
    parser.add_argument("--epochs", type=int, default=1, help="Total target epochs including previously completed epochs")
    parser.add_argument("--resume", help="Checkpoint folder or prior output folder with last_checkpoint.json")
    parser.add_argument("--save-every", type=int, default=100, help="Save optimizer state every N steps, plus epoch ends and final step")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--pilot", help="Verified joint pilot required for more than one optimizer step")
    parser.add_argument("--baseline-evaluation", help="Optional baseline validation_metrics.json; not required for training")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify_unified(args.manifest, args.output, args.visual_revision, args.audio_revision)
        print("Unified adapter/projector reload verified: max score difference %.6f" % result["max_score_difference"])
        return
    config = UnifiedConfig.load(args.config) if args.config else None
    train_unified(args.manifest, args.output, args.visual_revision, args.audio_revision, config,
                  None if args.max_steps == 0 else args.max_steps, args.epochs,
                  args.gradient_accumulation, args.learning_rate, args.seed, args.augment,
                  args.log_every, args.pilot, args.baseline_evaluation, args.resume, args.save_every)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    subprocess.run([sys.executable, "-u", "-m", "scripts.train_unified", "--verify-only",
                    "--manifest", args.manifest, "--output", args.output,
                    "--visual-revision", args.visual_revision, "--audio-revision", args.audio_revision], check=True)


if __name__ == "__main__":
    main()
