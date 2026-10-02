"""Joint embedding input to Nemotron's decoder; no late fusion or audio text proxy.

Torch is optional for the rest of the application; import this module only on
the training/inference path. The tiny-model tests exercise this same wrapper.
"""

import json
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from .labels import EXPRESSION_LABELS

INSTRUCTION = (
    "Use the matching face frames and speech audio to classify the presented expression. "
    "Reply with JSON containing only presented_expression and one of ANG, DIS, FEA, HAP, NEU, SAD."
)
MARKERS = ("__AV_FRAME_0__", "__AV_FRAME_1__", "__AV_FRAME_2__", "__AV_AUDIO__")


class AudioVisualModel(nn.Module):
    def __init__(self, visual, audio, tokenizer, processor, extractor, config):
        super().__init__()
        self.visual = visual
        self.audio = audio
        self.tokenizer = tokenizer
        self.processor = processor
        # The pinned custom processor reads these instance attributes. Its
        # Transformers fast-processor base filters them out as call-time kwargs.
        self.processor.max_num_tiles = 1
        self.processor.use_thumbnail = False
        self.extractor = extractor
        self.architecture = config.validate()
        hidden = visual.language_model.get_input_embeddings().weight.shape[1]
        self.audio_projector = nn.Sequential(
            nn.LayerNorm(audio.config.hidden_size),
            nn.Linear(audio.config.hidden_size, config.projector_hidden_size),
            nn.GELU(), nn.Linear(config.projector_hidden_size, hidden),
        ).to(device=self.device, dtype=torch.float32)
        for module in (self.audio, self.visual.vision_model, self.visual.mlp1):
            module.requires_grad_(False)
            module.eval()

    @property
    def device(self):
        return self.visual.language_model.get_input_embeddings().weight.device

    def train(self, mode=True):
        super().train(mode)
        # Frozen encoders must not acquire dropout or running-state updates.
        self.audio.eval()
        self.visual.vision_model.eval()
        self.visual.mlp1.eval()
        return self

    @torch.no_grad()
    def encode(self, images, samples):
        if len(images) != 3 or not samples:
            raise ValueError("joint input needs three frames and one waveform")
        frames = []
        side = math.isqrt(self.architecture.visual_tokens_per_frame)
        for image in images:
            # One crop tile per frame bounds GPU cost and preserves frame order.
            batch = self.processor([image], return_tensors="pt")
            pixels = batch["pixel_values"].to(device=self.device, dtype=torch.bfloat16)
            features = self.visual.extract_feature(pixels)
            if features.ndim != 3 or features.shape[0] != 1:
                raise ValueError("expected one visual tile per frame")
            grid = math.isqrt(features.shape[1])
            if grid * grid != features.shape[1] or side > grid:
                raise ValueError("unexpected Nemotron visual token grid")
            pooled = F.adaptive_avg_pool2d(
                features.reshape(1, grid, grid, -1).permute(0, 3, 1, 2).float(), (side, side))
            frames.append(pooled.flatten(2).transpose(1, 2).to(features.dtype))
        batch = self.extractor(samples, sampling_rate=16000, return_tensors="pt",
                               padding=False, return_attention_mask=True)
        inputs = {name: value.to(self.device) for name, value in batch.items()
                  if name in ("input_values", "attention_mask")}
        speech = self.audio(**inputs).last_hidden_state
        if "attention_mask" in inputs:
            mask = self.audio._get_feature_vector_attention_mask(speech.shape[1], inputs["attention_mask"])
            speech = speech[:, mask[0].bool(), :]
        if speech.shape[1] == 0:
            raise ValueError("audio encoder returned no valid time steps")
        speech = F.adaptive_avg_pool1d(speech.transpose(1, 2).float(),
                                      self.architecture.audio_tokens).transpose(1, 2)
        if not all(torch.isfinite(item).all() for item in frames + [speech]):
            raise ValueError("nonfinite encoder features")
        return torch.cat(frames, dim=1).detach(), speech.detach()

    def decoder_inputs(self, visual_features, audio_features, answer, face_mask=(True, True, True),
                       ablation=None):
        if answer not in EXPRESSION_LABELS or ablation not in (None, "audio", "video"):
            raise ValueError("invalid answer or ablation")
        if len(face_mask) != 3:
            raise ValueError("three face flags are required")
        embedding = self.visual.language_model.get_input_embeddings()
        count = self.architecture.visual_tokens_per_frame
        hidden = embedding.weight.shape[1]
        if tuple(visual_features.shape) != (1, 3 * count, hidden):
            raise ValueError("unexpected joint visual feature shape")
        if tuple(audio_features.shape) != (1, self.architecture.audio_tokens, self.audio.config.hidden_size):
            raise ValueError("unexpected joint audio feature shape")
        audio_tokens = self.audio_projector(audio_features.float()).to(embedding.weight.dtype)
        blocks = list(visual_features.split(count, dim=1)) + [audio_tokens]
        valid = [bool(flag) and ablation != "video" for flag in face_mask] + [ablation != "audio"]
        content = INSTRUCTION + "\n" + "\n".join(
            "Frame %d: <img>%s</img>" % (i + 1, MARKERS[i]) for i in range(3))
        content += "\nSpeech audio: " + MARKERS[3] + "\n"
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        pieces = re.split("(" + "|".join(MARKERS) + ")", prompt)
        if [piece for piece in pieces if piece in MARKERS] != list(MARKERS):
            raise ValueError("chat template changed the modality marker order")
        embeds, masks = [], []
        for piece in pieces:
            if piece in MARKERS:
                index = MARKERS.index(piece)
                block = blocks[index].to(device=self.device, dtype=embedding.weight.dtype)
                embeds.append(block if valid[index] else torch.zeros_like(block))
                masks.extend([int(valid[index])] * block.shape[1])
            elif piece:
                ids = self.tokenizer(piece, add_special_tokens=False)["input_ids"]
                embeds.append(embedding(torch.tensor([ids], device=self.device)))
                masks.extend([1] * len(ids))
        prompt_length = len(masks)
        completion = json.dumps({"presented_expression": answer}, separators=(",", ":"))
        if not self.tokenizer.eos_token:
            raise ValueError("tokenizer must define an end-of-sequence token")
        ids = self.tokenizer(completion + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        embeds.append(embedding(torch.tensor([ids], device=self.device)))
        masks.extend([1] * len(ids))
        mask = torch.tensor([masks], device=self.device, dtype=torch.long)
        labels = torch.tensor([[-100] * prompt_length + ids], device=self.device, dtype=torch.long)
        return {"inputs_embeds": torch.cat(embeds, dim=1), "attention_mask": mask,
                "position_ids": (mask.cumsum(-1) - 1).clamp_min(0), "labels": labels,
                "use_cache": False}

    def forward(self, visual_features, audio_features, answer, face_mask=(True, True, True), ablation=None):
        return self.visual.language_model(**self.decoder_inputs(
            visual_features, audio_features, answer, face_mask, ablation))

    @torch.no_grad()
    def scores(self, visual_features, audio_features, face_mask=(True, True, True), ablation=None):
        self.eval()
        likelihoods = []
        for label in EXPRESSION_LABELS:
            inputs = self.decoder_inputs(visual_features, audio_features, label, face_mask, ablation)
            loss = self.visual.language_model(**inputs).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite joint label likelihood")
            likelihoods.append(-float(loss) * int((inputs["labels"][:, 1:] != -100).sum()))
        probabilities = torch.softmax(torch.tensor(likelihoods, dtype=torch.float64), 0).tolist()
        return dict(zip(EXPRESSION_LABELS, probabilities))

    def gradient_report(self):
        groups = {"audio_projector": [], "decoder_lora": []}
        frozen_clean = True
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                frozen_clean &= parameter.grad is None
            elif name.startswith("audio_projector."):
                groups["audio_projector"].append(parameter)
            elif "lora_" in name:
                groups["decoder_lora"].append(parameter)
            else:
                raise RuntimeError("unexpected trainable parameter: " + name)
        report = {"frozen_encoders_have_no_gradients": bool(frozen_clean)}
        for name, parameters in groups.items():
            gradients = [p.grad for p in parameters if p.grad is not None]
            finite = bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)
            norm = sum(float(g.float().square().sum()) for g in gradients) ** 0.5
            report[name] = {"finite": finite, "norm": norm, "nonzero": finite and norm > 0}
        return report
