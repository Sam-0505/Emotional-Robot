"""Evaluate saved clip predictions with actor-disjoint validation and test sets.

Prediction JSONL rows require sample_id, scores, and optionally visual_quality
or audio_quality. Scores must come from model inference, never from labels.
"""

import argparse
import json
from pathlib import Path

from reachy_emotions.perception.fusion import (FusionConfig, fit_fusion,
                                                fit_temperature, fuse_predictions)
from reachy_emotions.perception.labels import UNKNOWN, best_label
from reachy_emotions.perception.manifest import read_manifest, resolve_media_path
from reachy_emotions.perception.metrics import evaluate_predictions


def _visual_frame_qualities(row):
    quality = row.get("frame_quality", [])
    if len(quality) != 3:
        raise ValueError("clip %s lacks three face quality records" % row["sample_id"])
    return [1.0 if isinstance(item, dict) and item.get("face_status") == "detected" else 0.0
            for item in quality]


def _read_predictions(path):
    predictions = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in predictions:
                raise ValueError("duplicate prediction for %s in %s" % (sample_id, path))
            predictions[sample_id] = row
    return predictions


def generate_predictions(manifest_path, output_dir, visual_revision, visual_adapter,
                         audio_revision, audio_head):
    """Run both pinned experts on the same validation and test clips."""
    from reachy_emotions.perception.visual import load_visual_expert, predict_visual_clip
    from reachy_emotions.perception.audio import load_audio_expert, predict_audio_clip

    records = [row for row in read_manifest(manifest_path)
               if row["split"] in ("validation", "test")]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    visual_path = out / "visual_predictions.jsonl"
    audio_path = out / "audio_predictions.jsonl"
    model, tokenizer, processor = load_visual_expert(visual_revision, visual_adapter)
    with visual_path.open("w", encoding="utf-8") as handle:
        for row in records:
            frame_paths = [resolve_media_path(manifest_path, path) for path in row["frame_paths"]]
            result = predict_visual_clip(model, tokenizer, processor, frame_paths,
                                         _visual_frame_qualities(row))
            handle.write(json.dumps({"sample_id": row["sample_id"], "model_id": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
                                     "revision": visual_revision,
                                     "adapter": str(visual_adapter) if visual_adapter else None,
                                     "mode": "fine_tuned" if visual_adapter else "zero_shot", **result}) + "\n")
    del model, tokenizer, processor
    encoder, head, extractor, device = load_audio_expert(audio_revision, audio_head)
    with audio_path.open("w", encoding="utf-8") as handle:
        for row in records:
            result = predict_audio_clip(encoder, head, extractor, device,
                                        resolve_media_path(manifest_path, row["audio_path"]))
            handle.write(json.dumps({"sample_id": row["sample_id"], "model_id": "microsoft/wavlm-base-plus",
                                     "revision": audio_revision, "head": str(audio_head), **result}) + "\n")
    return visual_path, audio_path


def generate_zero_shot_visual_predictions(manifest_path, output_dir, visual_revision):
    """Run V0 from the untouched checkpoint over the paired held-out clips."""
    from reachy_emotions.perception.visual import load_visual_expert, predict_visual_clip

    records = [row for row in read_manifest(manifest_path)
               if row["split"] in ("validation", "test")]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "visual_zero_shot_predictions.jsonl"
    model, tokenizer, processor = load_visual_expert(visual_revision)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            frames = [resolve_media_path(manifest_path, item) for item in row["frame_paths"]]
            result = predict_visual_clip(model, tokenizer, processor, frames,
                                         _visual_frame_qualities(row))
            handle.write(json.dumps({"sample_id": row["sample_id"], "model_id": "nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
                                     "revision": visual_revision, "adapter": None,
                                     "mode": "zero_shot", **result}) + "\n")
    return path


def evaluate(manifest_path, visual_path, audio_path, output_dir, zero_shot_visual_path=None):
    manifest = read_manifest(manifest_path)
    visual = _read_predictions(visual_path)
    audio = _read_predictions(audio_path)
    zero_shot = _read_predictions(zero_shot_visual_path) if zero_shot_visual_path else None
    expected_ids = {row["sample_id"] for row in manifest if row["split"] in ("validation", "test")}
    for name, predictions in (("visual", visual), ("audio", audio), ("zero-shot visual", zero_shot)):
        if predictions is None:
            continue
        missing, extra = expected_ids - set(predictions), set(predictions) - expected_ids
        if missing or extra:
            raise ValueError("%s prediction IDs do not match validation/test manifest: %d missing, %d extra"
                             % (name, len(missing), len(extra)))
    rows = []
    for record in manifest:
        if record["split"] not in ("validation", "test"):
            continue
        sample_id = record["sample_id"]
        if sample_id not in visual or sample_id not in audio:
            raise ValueError("paired prediction missing for %s" % sample_id)
        visual_row, audio_row = visual[sample_id], audio[sample_id]
        row = dict(record)
        row["visual_scores"] = visual_row.get("scores")
        row["audio_scores"] = audio_row.get("scores")
        row["visual_quality"] = visual_row.get("visual_quality", 1.0)
        row["audio_quality"] = audio_row.get("audio_quality", 1.0)
        if zero_shot is not None:
            if sample_id not in zero_shot:
                raise ValueError("zero-shot visual prediction missing for %s" % sample_id)
            row["zero_shot_visual_scores"] = zero_shot[sample_id].get("scores")
        rows.append(row)
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    if not validation or not test:
        raise ValueError("both validation and held-out test clips are required")
    visual_temperature = fit_temperature(validation, "visual_scores", "face_vote")
    audio_temperature = fit_temperature(validation, "audio_scores", "voice_vote")
    config = fit_fusion(validation, config=FusionConfig(visual_temperature=visual_temperature,
                                                       audio_temperature=audio_temperature))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.save(out / "fusion_config.json")
    validation_scored, validation_metrics = _score_partition(validation, config, zero_shot is not None)
    scored, metrics = _score_partition(test, config, zero_shot is not None)
    metrics.update({
        "validation_selection": validation_metrics,
        "provenance": {"manifest": str(manifest_path), "visual_predictions": str(visual_path),
                       "audio_predictions": str(audio_path), "fusion_config": str(out / "fusion_config.json"),
                       "selection_split": "validation", "evaluation_split": "test"},
    })
    if zero_shot is not None:
        metrics["visual_face_vote_macro_f1_gain_over_zero_shot"] = (
            metrics["visual_face_vote"]["macro_f1"] - metrics["visual_zero_shot_face_vote"]["macro_f1"])
        metrics["provenance"]["zero_shot_visual_predictions"] = str(zero_shot_visual_path)
    (out / "validation_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in validation_scored), encoding="utf-8")
    (out / "validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8")
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _score_partition(rows, config, include_zero_shot):
    scored = []
    for row in rows:
        result = fuse_predictions(row["visual_scores"], row["audio_scores"],
                                  row["visual_quality"], row["audio_quality"], config)
        prediction = {
            "sample_id": row["sample_id"], "actor_id": row["actor_id"],
            "face_vote": row["face_vote"], "voice_vote": row["voice_vote"],
            "multimodal_vote": row["multimodal_vote"],
            "visual_prediction": best_label(row["visual_scores"]) if row["visual_scores"] else UNKNOWN,
            "audio_prediction": best_label(row["audio_scores"]) if row["audio_scores"] else UNKNOWN,
            "fused_prediction": result["presented_expression"], "fusion": result,
        }
        if include_zero_shot:
            scores = row["zero_shot_visual_scores"]
            prediction["zero_shot_visual_prediction"] = best_label(scores) if scores else UNKNOWN
        scored.append(prediction)
    metrics = {
        "visual_face_vote": evaluate_predictions(scored, "face_vote", "visual_prediction"),
        "audio_voice_vote": evaluate_predictions(scored, "voice_vote", "audio_prediction"),
        "fusion_multimodal_vote": evaluate_predictions(scored, "multimodal_vote", "fused_prediction"),
        "visual_multimodal_vote": evaluate_predictions(scored, "multimodal_vote", "visual_prediction"),
        "audio_multimodal_vote": evaluate_predictions(scored, "multimodal_vote", "audio_prediction"),
    }
    if include_zero_shot:
        metrics["visual_zero_shot_face_vote"] = evaluate_predictions(
            scored, "face_vote", "zero_shot_visual_prediction")
        metrics["visual_zero_shot_multimodal_vote"] = evaluate_predictions(
            scored, "multimodal_vote", "zero_shot_visual_prediction")
    return scored, metrics


def main():
    parser = argparse.ArgumentParser(description="Fit validation calibration and evaluate paired test clips")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--visual-predictions", help="Existing visual score JSONL")
    parser.add_argument("--audio-predictions", help="Existing audio score JSONL")
    parser.add_argument("--visual-revision", help="Pinned NVIDIA revision for live inference")
    parser.add_argument("--visual-adapter", help="Saved visual LoRA adapter")
    parser.add_argument("--audio-revision", help="Pinned WavLM revision for live inference")
    parser.add_argument("--audio-head", help="Saved audio classifier head")
    parser.add_argument("--zero-shot-visual", action="store_true",
                        help="Score the untouched NVIDIA checkpoint on the same clips")
    parser.add_argument("--zero-shot-visual-predictions",
                        help="Existing V0 prediction JSONL; avoids another model load")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.visual_predictions) != bool(args.audio_predictions):
        parser.error("provide both prediction files or neither")
    if args.zero_shot_visual and args.zero_shot_visual_predictions:
        parser.error("choose generated or existing zero-shot predictions")
    if args.visual_predictions:
        visual_path, audio_path = args.visual_predictions, args.audio_predictions
    else:
        if not all((args.visual_revision, args.audio_revision, args.audio_head)):
            parser.error("live inference requires both revisions and an audio head; visual adapter is optional for V0")
        visual_path, audio_path = generate_predictions(args.manifest, args.output,
                                                       args.visual_revision, args.visual_adapter,
                                                       args.audio_revision, args.audio_head)
    zero_shot_path = args.zero_shot_visual_predictions
    if args.zero_shot_visual:
        if not args.visual_revision:
            parser.error("--zero-shot-visual requires --visual-revision")
        zero_shot_path = generate_zero_shot_visual_predictions(args.manifest, args.output,
                                                               args.visual_revision)
    metrics = evaluate(args.manifest, visual_path, audio_path, args.output, zero_shot_path)
    print(json.dumps({"test_clips": metrics["fusion_multimodal_vote"]["clips"],
                      "fusion_macro_f1": metrics["fusion_multimodal_vote"]["macro_f1"],
                      "fusion_accuracy": metrics["fusion_multimodal_vote"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
