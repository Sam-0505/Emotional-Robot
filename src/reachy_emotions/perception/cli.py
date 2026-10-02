"""Shared selection of unified perception or retained late-fusion baselines."""

from pathlib import Path


def add_perception_arguments(parser):
    parser.add_argument("--visual-revision", help="Pinned NVIDIA checkpoint commit")
    parser.add_argument("--audio-revision", help="Pinned WavLM checkpoint commit")
    parser.add_argument("--visual-adapter", type=Path)
    parser.add_argument("--audio-head", type=Path)
    parser.add_argument("--fusion-config", type=Path)
    parser.add_argument("--unified-checkpoint", type=Path, help="Joint decoder LoRA plus audio projector directory")
    parser.add_argument("--unified-calibration", type=Path, help="Validation-fitted unified calibration JSON")


def validate_perception_arguments(parser, args, require_baseline_calibration=False):
    if args.unified_checkpoint:
        if not args.unified_calibration:
            parser.error("--unified-checkpoint requires --unified-calibration")
        if args.visual_adapter or args.audio_head or args.fusion_config:
            parser.error("do not mix unified checkpoints with baseline adapter/head/fusion arguments")
    else:
        if args.unified_calibration:
            parser.error("--unified-calibration requires --unified-checkpoint")
        if not all((args.visual_revision, args.audio_revision, args.audio_head)):
            parser.error("baseline perception requires both revisions and --audio-head")
        if require_baseline_calibration and not args.fusion_config:
            parser.error("integrated baseline perception requires --fusion-config")


def infer_from_arguments(args, record):
    if args.unified_checkpoint:
        from .unified import checkpoint_identity, checkpoint_metadata, file_hash, load_unified
        from .unified_calibration import UnifiedCalibration
        from ..pipeline import infer_unified_manifest_record
        metadata = checkpoint_metadata(args.unified_checkpoint, args.visual_revision, args.audio_revision)
        if not metadata.get("reload_verified"):
            raise ValueError("unified checkpoint must pass reload verification before integration")
        manifest_hash = file_hash(args.manifest)
        if metadata["manifest_sha256"] != manifest_hash:
            raise ValueError("unified manifest differs from training actor splits")
        config = UnifiedCalibration.load(args.unified_calibration)
        config.check(checkpoint_identity(metadata), manifest_hash)
        model = load_unified(metadata["visual_revision"], metadata["audio_revision"],
                             checkpoint=args.unified_checkpoint)
        return infer_unified_manifest_record(args.manifest, record, model, config)
    from .audio import load_audio_expert
    from .visual import load_visual_expert
    from .fusion import FusionConfig
    from ..pipeline import infer_manifest_record
    visual = load_visual_expert(args.visual_revision, args.visual_adapter)
    audio = load_audio_expert(args.audio_revision, args.audio_head)
    config = FusionConfig.load(args.fusion_config) if args.fusion_config else FusionConfig()
    return infer_manifest_record(args.manifest, record, visual, audio, config)
