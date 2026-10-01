"""Run Nemotron FaceVote LoRA feasibility: python -m scripts.train_visual ..."""

import argparse
import gc
import subprocess
import sys

from reachy_emotions.perception.visual import train_visual_adapter, verify_visual_adapter


def main():
    parser = argparse.ArgumentParser(description="Train decoder-only LoRA on CREMA-D FaceVote")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True, help="Pinned NVIDIA model revision")
    parser.add_argument("--max-steps", type=int, default=1,
                        help="Optimizer steps; defaults to one feasibility step. Use 0 for all epochs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--augment", action="store_true", help="Training-only framing, mirror and photometric variation")
    parser.add_argument("--verify-only", action="store_true", help="Reload an existing adapter and compare label scores")
    args = parser.parse_args()
    if args.verify_only:
        verification = verify_visual_adapter(args.manifest, args.output, args.revision)
        print("Visual adapter reload verified: max score difference %.6f" % verification["max_score_difference"])
        return
    output = train_visual_adapter(args.manifest, args.output, args.revision,
                                  None if args.max_steps == 0 else args.max_steps,
                                  args.learning_rate, args.epochs,
                                  args.gradient_accumulation, args.seed, args.augment)
    print("Saved visual adapter and provenance to %s" % output)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    subprocess.run([sys.executable, "-m", "scripts.train_visual", "--verify-only",
                    "--manifest", args.manifest, "--output", args.output,
                    "--revision", args.revision], check=True)


if __name__ == "__main__":
    main()
