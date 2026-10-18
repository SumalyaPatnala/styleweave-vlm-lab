"""Load frozen OpenAI CLIP. Optional Hugging Face download is gated."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from styleweave.preflight import check_disk_budget, detect_runtime


class FrozenCLIP:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str,
        *,
        allow_download: bool,
        min_free_gib: float,
        cache_dir: str | None = None,
        frozen: bool = True,
        skip_disk_check: bool = False,
    ) -> None:
        if not frozen:
            raise ValueError("Phase 1 B0 requires frozen=True")
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required to run B0 encoding. "
                "Install Phase 1 extras, or run on Kaggle where they are present."
            ) from exc

        if allow_download and not skip_disk_check:
            check_disk_budget(
                runtime=detect_runtime(),
                cache_dir=cache_dir,
                allow_download=True,
                model_name=model_name,
                mac_min_free_gib=min_free_gib,
            )

        kwargs: dict = {
            "revision": revision,
            "local_files_only": not allow_download,
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        try:
            self.model = CLIPModel.from_pretrained(model_name, **kwargs)
            self.processor = CLIPProcessor.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            if not allow_download:
                raise RuntimeError(
                    "CLIP weights are not in the local cache. Re-run with "
                    "--allow-download only after a disk check, preferably on Kaggle."
                ) from exc
            raise

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = device
        self.torch = torch
        self.model.to(device)
        self.model_name = model_name
        self.revision_requested = revision
        self.revision_resolved = getattr(self.model.config, "_commit_hash", revision)

    def trainable_module_names(self) -> list[str]:
        return [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

    def encode_images(self, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        from PIL import Image

        features: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                images = [Image.open(path).convert("RGB") for path in batch_paths]
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                batch = self.model.get_image_features(pixel_values=pixel_values)
                features.append(batch.detach().cpu().float().numpy())
        if not features:
            return np.zeros((0, 512), dtype=np.float32)
        return np.concatenate(features, axis=0)

    def encode_texts(
        self, texts: Sequence[str], batch_size: int = 32, max_length: int = 77
    ) -> np.ndarray:
        features: list[np.ndarray] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_text = list(texts[start : start + batch_size])
                inputs = self.processor(
                    text=batch_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                kwargs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                    if key in {"input_ids", "attention_mask"} and hasattr(value, "to")
                }
                batch = self.model.get_text_features(**kwargs)
                features.append(batch.detach().cpu().float().numpy())
        if not features:
            return np.zeros((0, 512), dtype=np.float32)
        return np.concatenate(features, axis=0)

    def card(self) -> dict:
        return {
            "name": self.model_name,
            "revision_requested": self.revision_requested,
            "revision_resolved": self.revision_resolved,
            "frozen": True,
            "trainable_modules": self.trainable_module_names(),
            "unconstrained_deployment": "out of scope per CLIP model card",
        }
