"""Train the CREMA-D VoiceVote head: python -m scripts.train_audio ..."""

import argparse
import gc
import subprocess
import sys

from reachy_emotions.perception.audio import train_audio_head, verify_audio_head


def main():
    parser = argparse.ArgumentParser(description="Train frozen-WavLM VoiceVote classifier")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True, help="Pinned WavLM Hugging Face revision")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Use 1 for save/reload feasibility gate")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="Training-only mild gain, noise and bandwidth variation")
    parser.add_argument("--verify-only", action="store_true", help="Reload a saved head and compare logits")
    args = parser.parse_args()
    if args.verify_only:
        verification = verify_audio_head(args.manifest, args.output, args.revision, device=args.device)
        print("Audio head reload verified: max logit difference %.6f" % verification["max_logit_difference"])
        return
    output = train_audio_head(args.manifest, args.output, args.revision, args.epochs,
                              args.max_steps, args.learning_rate, args.device,
                              args.seed, args.augment)
    print("Saved audio head and provenance to %s" % output)
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    command = [sys.executable, "-m", "scripts.train_audio", "--verify-only",
               "--manifest", args.manifest, "--output", args.output, "--revision", args.revision]
    if args.device:
        command.extend(["--device", args.device])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
