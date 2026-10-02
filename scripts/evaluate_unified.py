"""Generate joint held-out scores; select on validation, then explicitly evaluate test."""

import argparse
import json
import time
from pathlib import Path

from reachy_emotions.perception.manifest import read_manifest
from reachy_emotions.perception.metrics import evaluate_predictions
from reachy_emotions.perception.unified import (checkpoint_identity, checkpoint_metadata, file_hash,
                                               load_unified, predict_unified_record)
from reachy_emotions.perception.unified_calibration import (UnifiedCalibration, calibrated_prediction,
                                                           fit_unified_calibration)
from reachy_emotions.perception.unified_config import clip_quality
from scripts.evaluate_perception import _read_predictions


def evaluate(manifest, predictions, output, checkpoint, split="validation", calibration_path=None,
             baseline_predictions=None):
    if split not in ("validation", "test"):
        raise ValueError("evaluate only validation or test actors")
    records = [row for row in read_manifest(manifest) if row["split"] == split]
    scores = _read_predictions(predictions)
    if set(scores) != {row["sample_id"] for row in records}:
        raise ValueError("unified prediction IDs must exactly match the selected split")
    metadata = checkpoint_metadata(checkpoint)
    if not metadata.get("reload_verified"):
        raise ValueError("checkpoint must pass fresh-process reload verification")
    identity, manifest_hash = checkpoint_identity(metadata), file_hash(manifest)
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("evaluation manifest differs from training actor splits")
    rows = []
    for row in records:
        prediction = scores[row["sample_id"]]
        if prediction.get("checkpoint_id") != identity or prediction.get("manifest_sha256") != manifest_hash:
            raise ValueError("unified predictions have stale checkpoint/manifest provenance")
        rows.append({**row, "scores": prediction.get("scores")})
    if calibration_path:
        config = UnifiedCalibration.load(calibration_path)
        config.check(identity, manifest_hash)
    elif split == "validation":
        config = fit_unified_calibration(rows, identity, manifest_hash)
    else:
        raise ValueError("test evaluation requires saved validation calibration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if split == "validation":
        config.save(output / "unified_calibration.json")
    scored = [{**row, **calibrated_prediction(row["scores"], config, clip_quality(row)["accepted"])} for row in rows]
    metrics = {"unified_multimodal_vote": evaluate_predictions(scored, "multimodal_vote"),
               "split": split, "checkpoint_id": identity, "manifest_sha256": manifest_hash,
               "json_output": "constrained_label_likelihoods_rendered_as_JSON; not free-form generation"}
    latencies = sorted(float(item["latency_seconds"]) for item in scores.values() if "latency_seconds" in item)
    if latencies:
        metrics["latency_seconds"] = {"p50": latencies[(len(latencies) - 1) // 2],
                                      "p95": latencies[min(len(latencies) - 1, int(len(latencies) * .95))]}
    for modality in ("audio", "video"):
        key = modality + "_ablated_scores"
        if all(key in prediction for prediction in scores.values() if prediction.get("scores") is not None):
            ablated = [{**row, **calibrated_prediction(scores[row["sample_id"]].get(key), config,
                                                       clip_quality(row)["accepted"])} for row in rows]
            metrics[modality + "_ablated_multimodal_vote"] = evaluate_predictions(ablated, "multimodal_vote")
    if baseline_predictions:
        baseline = _read_predictions(baseline_predictions)
        if set(baseline) != set(scores):
            raise ValueError("baseline comparison must use exactly the same split clips")
        for key in ("visual_prediction", "audio_prediction", "fused_prediction"):
            compared = [{**row, "prediction": baseline[row["sample_id"]][key]} for row in rows]
            metrics[key + "_multimodal_vote"] = evaluate_predictions(compared, "multimodal_vote", "prediction")
    (output / (split + "_metrics.json")).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / (split + "_predictions.jsonl")).write_text(
        "".join(json.dumps(row) + "\n" for row in scored), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--calibration", help="Required for test; fitted on validation only")
    parser.add_argument("--predictions", help="Existing joint predictions for selected split")
    parser.add_argument("--baseline-predictions", help="Baseline evaluation's matching split predictions JSONL")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.split == "test" and not args.calibration:
        parser.error("--split test requires --calibration from validation")
    metadata = checkpoint_metadata(args.checkpoint)
    if not metadata.get("reload_verified"):
        parser.error("checkpoint must pass fresh-process reload verification")
    manifest_hash = file_hash(args.manifest)
    if metadata["manifest_sha256"] != manifest_hash:
        parser.error("evaluation manifest differs from training actor splits")
    if args.calibration:
        UnifiedCalibration.load(args.calibration).check(checkpoint_identity(metadata), manifest_hash)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("evaluation output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    path = args.predictions
    if not path:
        model = load_unified(metadata["visual_revision"], metadata["audio_revision"], checkpoint=args.checkpoint)
        identity = checkpoint_identity(metadata)
        path = output / "unified_scores.jsonl"
        rows = [row for row in read_manifest(args.manifest) if row["split"] == args.split]
        with path.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                started = time.perf_counter()
                result = predict_unified_record(model, args.manifest, row, args.ablations)
                handle.write(json.dumps({"sample_id": row["sample_id"], "checkpoint_id": identity,
                                         "manifest_sha256": manifest_hash,
                                         "latency_seconds": time.perf_counter() - started, **result}) + "\n")
                if index == 1 or index % 25 == 0 or index == len(rows):
                    print("Unified %s %d/%d" % (args.split, index, len(rows)), flush=True)
    print(json.dumps(evaluate(args.manifest, path, output, args.checkpoint, args.split,
                              args.calibration, args.baseline_predictions), indent=2))


if __name__ == "__main__":
    main()
