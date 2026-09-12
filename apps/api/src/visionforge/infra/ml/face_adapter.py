"""Face detection.

**Detection only. No recognition, no identity, no embeddings of faces.** The
output is a count and boxes; nothing stored here can identify a person, and
nothing downstream is given the means to.

Model choice, and a deliberate deviation from the Phase 3 brief
----------------------------------------------------------------
The brief preferred SCRFD. This uses **YuNet**, which comes from the same family
of lightweight anchor-based detectors and ships in the OpenCV Zoo, for three
reasons specific to this machine:

- **232 KB**, versus SCRFD's multi-megabyte weights.
- **No new runtime.** SCRFD in practice means ``insightface`` plus
  ``onnxruntime-gpu``, and no onnxruntime-gpu build targets CUDA 13 yet. OpenCV's
  DNN module already loads YuNet's ONNX graph.
- **It runs on CPU in single-digit milliseconds**, leaving the whole 4 GiB card
  to CLIP. On this hardware, spending VRAM on face detection would be the wrong
  trade.

The ``Analyzer`` port means swapping in a GPU SCRFD later is a new adapter and a
config change, not a refactor -- which is the point of having the port.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2

from visionforge.domain.analysis import (
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
)
from visionforge.domain.errors import TransientError
from visionforge.domain.media import MediaKind
from visionforge.infra.analysis.frames import sample_frames

logger = logging.getLogger(__name__)

FACE_ANALYZER_VERSION = "1"
FACE_MODEL_NAME = "yunet-2023mar"

#: Pinned release asset plus its checksum. A model fetched over the network is
#: only trustworthy if its bytes are verified.
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

#: Confidence floor. 0.6 rather than the 0.9 a demo would use: a missed face in a
#: wide shot costs more here than an occasional false positive, because the count
#: feeds selection heuristics rather than anything user-visible.
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 500


def model_cache_dir() -> Path:
    """Where weights live: outside the repository, per the Phase 0 rule.

    Never committed, never inside a Docker build context, and shared between
    projects on this machine.
    """
    root = os.environ.get("VF_MODEL_CACHE") or os.environ.get("TORCH_HOME")
    base = Path(root) if root else Path.home() / ".cache" / "visionforge"
    path = base / "opencv"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_yunet_model() -> Path:
    """Return the local model path, downloading and verifying it once."""
    import hashlib

    destination = model_cache_dir() / YUNET_FILENAME
    if destination.exists():
        return destination

    logger.info("downloading face detection model", extra={"url": YUNET_URL})
    try:
        with urllib.request.urlopen(YUNET_URL, timeout=120) as response:
            payload = response.read()
    except Exception as exc:
        raise TransientError(f"could not download the face detection model: {exc}") from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest != YUNET_SHA256:
        raise TransientError(
            "face detection model failed its checksum",
            hint=f"expected {YUNET_SHA256}, got {digest}",
        )

    destination.write_bytes(payload)
    return destination


class LoadedYuNet:
    """An OpenCV FaceDetectorYN instance."""

    def __init__(self, detector: Any) -> None:
        self.detector = detector

    def unload(self) -> None:
        del self.detector


class YuNetModelSpec:
    key = "faces:yunet"
    version = FACE_ANALYZER_VERSION
    #: Zero: the graph runs on CPU, so it consumes no VRAM and never competes
    #: with CLIP for the budget.
    estimated_vram_mb = 0

    def load(self) -> LoadedYuNet:
        path = ensure_yunet_model()
        detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            top_k=TOP_K,
        )
        return LoadedYuNet(detector)


class FaceDetectionAnalyzer:
    """Counts faces and reports their boxes. Never who they are."""

    name = AnalyzerName.FACES
    version = FACE_ANALYZER_VERSION
    kind = AnalyzerKind.GPU  # runs in the GPU job alongside CLIP; executes on CPU

    def __init__(self) -> None:
        self._loaded: LoadedYuNet | None = None

    def _detector(self) -> LoadedYuNet:
        if self._loaded is None:
            self._loaded = YuNetModelSpec().load()
        return self._loaded

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind in (MediaKind.IMAGE, MediaKind.VIDEO)

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"face detection does not apply to {source.kind.value}"},
            )

        detector = self._detector().detector
        frames = sample_frames(source)
        per_frame: list[dict[str, Any]] = []

        for frame in frames:
            height, width = frame.image.shape[:2]
            detector.setInputSize((width, height))
            _, detections = detector.detect(frame.image)

            faces = []
            if detections is not None:
                for row in detections:
                    x, y, box_w, box_h = (float(v) for v in row[:4])
                    faces.append(
                        {
                            # Normalised to 0-1 so boxes stay valid regardless of
                            # whether the proxy or the original was analysed.
                            "x": round(x / width, 5),
                            "y": round(y / height, 5),
                            "width": round(box_w / width, 5),
                            "height": round(box_h / height, 5),
                            "confidence": round(float(row[-1]), 4),
                        }
                    )

            per_frame.append(
                {
                    "timestamp_ms": frame.timestamp_ms,
                    "face_count": len(faces),
                    "faces": faces,
                }
            )

        counts = [int(f["face_count"]) for f in per_frame]
        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            payload={
                "model": FACE_MODEL_NAME,
                "detector": "yunet",
                "device": "cpu",
                # Max rather than mean: a clip that shows a face in any sampled
                # frame contains a face, even if most frames are scenery.
                "face_count": max(counts) if counts else 0,
                "frames_with_faces": sum(1 for c in counts if c > 0),
                "frame_count": len(per_frame),
                "frames": per_frame,
                "score_threshold": SCORE_THRESHOLD,
                "identity_stored": False,
                "used_proxy": source.used_proxy,
            },
        )
