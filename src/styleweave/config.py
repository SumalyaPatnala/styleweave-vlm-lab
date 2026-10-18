"""Experiment configuration. YAML is optional; defaults match Phase 0 B0."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "experiments" / "phase1_frozen_clip.yaml"

CATEGORIES = ("dress", "shirt", "toptee")
CLIP_ID = "openai/clip-vit-base-patch32"
# Hugging Face commit verified by HTTP HEAD in Phase 0 (2026-10-18).
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MIN_FREE_GIB = 25.0


@dataclass
class HubConfig:
    enabled: bool = False
    require_env: str = "STYLEWEAVE_HF_PUBLISH"
    repo_id: str = ""
    private: bool = True
    allow_images: bool = False


@dataclass
class B0Config:
    experiment_id: str = "phase1_frozen_clip"
    seed: int = 42
    device_preference: list[str] = field(
        default_factory=lambda: ["cuda", "mps", "cpu"]
    )
    model_name: str = CLIP_ID
    model_revision: str = CLIP_REVISION
    frozen: bool = True
    query_mode: str = "l2_sum"
    caption_mode: str = "clip4cir_concat"
    report_split: str = "val"
    categories: tuple[str, ...] = CATEGORIES
    captions_dir: str = "data/raw/fashion-iq/captions"
    image_splits_dir: str = "data/raw/fashion-iq/image_splits"
    images_dir: str = "data/raw/fashion-iq/images"
    allow_missing_gallery_frac: float = 0.0
    fail_on_missing_target: bool = True
    fail_on_missing_reference: bool = True
    require_complete_gallery: bool = True
    incomplete_gallery_justification: str = ""
    batch_size: int = 32
    max_text_length: int = 77
    artifacts_dir: str = "artifacts/phase1_frozen_clip"
    allow_download: bool = False
    min_free_gib: float = MIN_FREE_GIB
    hub: HubConfig = field(default_factory=HubConfig)
    literature_source: str = "CLIP4Cir Table 1 frozen sum (Baldrati et al. 2023)"
    literature_r10: float = 19.70
    literature_r50: float = 38.26

    def resolve(self, root: Path | None = None) -> "B0Config":
        base = root or ROOT
        self.captions_dir = str((base / self.captions_dir).resolve()) if not Path(self.captions_dir).is_absolute() else self.captions_dir
        self.image_splits_dir = str((base / self.image_splits_dir).resolve()) if not Path(self.image_splits_dir).is_absolute() else self.image_splits_dir
        self.images_dir = str((base / self.images_dir).resolve()) if not Path(self.images_dir).is_absolute() else self.images_dir
        self.artifacts_dir = str((base / self.artifacts_dir).resolve()) if not Path(self.artifacts_dir).is_absolute() else self.artifacts_dir
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load experiment YAML") from exc
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return data


def load_b0_config(path: Path | None = None, root: Path | None = None) -> B0Config:
    cfg = B0Config()
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.is_file():
        raw = _load_yaml(config_path)
        cfg.experiment_id = raw.get("experiment_id", cfg.experiment_id)
        cfg.seed = int(raw.get("seed", cfg.seed))
        cfg.device_preference = list(raw.get("device_preference", cfg.device_preference))
        model = raw.get("model", {})
        cfg.model_name = model.get("name", cfg.model_name)
        cfg.model_revision = str(model.get("revision", cfg.model_revision))
        cfg.frozen = bool(model.get("frozen", cfg.frozen))
        query = raw.get("query", {})
        cfg.query_mode = query.get("primary", cfg.query_mode)
        cfg.caption_mode = query.get("caption_mode", cfg.caption_mode)
        data = raw.get("data", {})
        cfg.report_split = data.get("report_split", cfg.report_split)
        cfg.categories = tuple(data.get("categories", cfg.categories))
        cfg.captions_dir = data.get("captions_dir", cfg.captions_dir)
        cfg.image_splits_dir = data.get("image_splits_dir", cfg.image_splits_dir)
        cfg.images_dir = data.get("images_dir", cfg.images_dir)
        cfg.allow_missing_gallery_frac = float(
            data.get("allow_missing_gallery_frac", cfg.allow_missing_gallery_frac)
        )
        cfg.fail_on_missing_target = bool(
            data.get("fail_on_missing_target", cfg.fail_on_missing_target)
        )
        cfg.fail_on_missing_reference = bool(
            data.get("fail_on_missing_reference", cfg.fail_on_missing_reference)
        )
        cfg.require_complete_gallery = bool(
            data.get("require_complete_gallery", cfg.require_complete_gallery)
        )
        cfg.incomplete_gallery_justification = str(
            data.get("incomplete_gallery_justification", cfg.incomplete_gallery_justification)
        )
        runtime = raw.get("runtime", {})
        cfg.batch_size = int(runtime.get("batch_size", cfg.batch_size))
        cfg.allow_download = bool(runtime.get("allow_download", cfg.allow_download))
        cfg.min_free_gib = float(runtime.get("min_free_gib", cfg.min_free_gib))
        artifacts = raw.get("artifacts", {})
        if "dir" in artifacts:
            cfg.artifacts_dir = artifacts["dir"]
        hub = raw.get("hub", {})
        cfg.hub = HubConfig(
            enabled=bool(hub.get("enabled", False)),
            require_env=str(hub.get("require_env", "STYLEWEAVE_HF_PUBLISH")),
            repo_id=str(hub.get("repo_id", "")),
            private=bool(hub.get("private", True)),
            allow_images=False,
        )
        lit = raw.get("literature_orientation", {})
        cfg.literature_source = lit.get("source", cfg.literature_source)
        cfg.literature_r10 = float(lit.get("mean_recall_at_10", cfg.literature_r10))
        cfg.literature_r50 = float(lit.get("mean_recall_at_50", cfg.literature_r50))
    return cfg.resolve(root=root)
