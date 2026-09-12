"""OpenCLIP image embeddings.

**ViT-B/32, not something larger.** Measured on this RTX 3050: 352 MiB resident
in fp16, 584 MiB peak on a batch of 8, 512-dimensional output. ViT-L/14 would
roughly quadruple that for a gain on retrieval benchmarks that nothing in
VisionForge has yet shown it needs. On a 4 GiB card, a model that leaves room for
a second one is worth more than a marginally better score.

The model name appears only in this file. Everything upstream sees an
``Analyzer``, which is what makes swapping to SigLIP or a hosted embedding API a
configuration change rather than a refactor.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import open_clip
from numpy.typing import NDArray
import torch
from PIL import Image

from visionforge.domain.analysis import (
    CLIP_EMBEDDING_DIM,
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
)
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.frames import sample_frames
from visionforge.infra.ml.device import select_device, use_fp16
from visionforge.infra.ml.registry import ModelRegistry, get_registry

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_ANALYZER_VERSION = "1"
CLIP_MODEL_KEY = "clip:vit-b-32"

#: Measured 584 MiB peak at batch 8; 900 leaves headroom for the allocator's
#: fragmentation and a transient cuDNN workspace without over-reserving the card.
CLIP_ESTIMATED_VRAM_MB = 900

#: Frames per forward pass. Five sampled frames fit in one batch, so a video
#: costs a single GPU round trip.
CLIP_BATCH_SIZE = 8


class LoadedClip:
    """A loaded OpenCLIP model plus its preprocessing transform."""

    def __init__(self, model: Any, preprocess: Any, device: str, fp16: bool) -> None:
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

    def unload(self) -> None:
        del self.model
        del self.preprocess


class ClipModelSpec:
    """Tells the registry how to build the CLIP model, and how big it is."""

    key = CLIP_MODEL_KEY
    version = CLIP_ANALYZER_VERSION
    estimated_vram_mb = CLIP_ESTIMATED_VRAM_MB

    def load(self) -> LoadedClip:
        device = select_device()
        fp16 = use_fp16()
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            device=device,
            precision="fp16" if fp16 else "fp32",
        )
        model.eval()
        return LoadedClip(model=model, preprocess=preprocess, device=device, fp16=fp16)


class ClipEmbeddingAnalyzer:
    """CLIP embeddings for images and representative video frames.

    A video is embedded by averaging its sampled frames. Averaging normalised
    embeddings gives a single vector describing the clip as a whole, which is
    what a "find clips like this" query needs; per-frame vectors are a Phase 4
    concern once scene-level retrieval is actually being asked for.
    """

    name = AnalyzerName.CLIP
    version = CLIP_ANALYZER_VERSION
    kind = AnalyzerKind.GPU

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or get_registry()
        self._registry.register(ClipModelSpec())

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind in (MediaKind.IMAGE, MediaKind.VIDEO)

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"CLIP embedding does not apply to {source.kind.value}"},
            )

        frames = sample_frames(source)
        lease = self._registry.lease(CLIP_MODEL_KEY)
        with lease as loaded:
            assert isinstance(loaded, LoadedClip)
            embedding, per_frame = self._embed(loaded, frames)

        receipt = lease.receipt
        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            embedding=tuple(float(v) for v in embedding),
            payload={
                "model": CLIP_MODEL_NAME,
                "pretrained": CLIP_PRETRAINED,
                "dim": CLIP_EMBEDDING_DIM,
                "frame_count": len(frames),
                "frame_timestamps_ms": [f.timestamp_ms for f in frames],
                "per_frame_embeddings": per_frame,
                "normalized": True,
                "used_proxy": source.used_proxy,
            },
            metrics=receipt.as_dict() if receipt else {},
        )

    def _embed(self, loaded: LoadedClip, frames: list[Any]) -> tuple[NDArray[np.float32], int]:
        """Encode frames and return one L2-normalised mean vector.

        Normalising *before* averaging, then again after, keeps every frame's
        contribution equal regardless of its raw magnitude -- otherwise a single
        high-norm frame would dominate the clip's vector.
        """
        tensors = [
            loaded.preprocess(Image.fromarray(frame.image[:, :, ::-1]))  # BGR -> RGB
            for frame in frames
        ]

        vectors: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(tensors), CLIP_BATCH_SIZE):
                batch = torch.stack(tensors[start : start + CLIP_BATCH_SIZE]).to(
                    loaded.device, dtype=loaded.dtype
                )
                features = loaded.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                vectors.append(features.float().cpu())

        stacked = torch.cat(vectors, dim=0)
        mean = stacked.mean(dim=0)
        mean = mean / mean.norm()
        return mean.numpy(), int(stacked.shape[0])


def embed_text(query: str, registry: ModelRegistry | None = None) -> tuple[float, ...]:
    """Embed a text query into the same space as the image embeddings.

    Present because it costs a few lines and makes the pgvector storage provably
    useful -- "find clips related to football" resolves to a vector search
    against this. **No search UI is built in Phase 3**; this is the retrieval
    foundation the API exposes and nothing more.
    """
    reg = registry or get_registry()
    reg.register(ClipModelSpec())

    with reg.lease(CLIP_MODEL_KEY) as loaded:
        assert isinstance(loaded, LoadedClip)
        tokens = open_clip.tokenize([query]).to(loaded.device)
        with torch.no_grad():
            features = loaded.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        vector = features.float().cpu().numpy()[0]
    return tuple(float(v) for v in vector)
