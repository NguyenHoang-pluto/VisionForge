"""Scene boundary detection.

Wraps PySceneDetect's content detector, which compares consecutive frames in HSV
and cuts where the difference exceeds a threshold. Deterministic: the same file
produces the same boundaries every time.

**The original is never touched.** Detection reads the 720p proxy where one
exists (see ``workers.media_analysis``), and it only reads -- no frames are
written, no file is modified.
"""

from __future__ import annotations

import logging

from scenedetect import ContentDetector, SceneManager, open_video

from visionforge.domain.analysis import (
    AnalysisOutcome,
    AnalysisSource,
    AnalysisStatus,
    AnalyzerKind,
    AnalyzerName,
)
from visionforge.domain.errors import UnsupportedMediaError
from visionforge.domain.media import MediaKind

logger = logging.getLogger(__name__)

SCENES_ANALYZER_VERSION = "1"

#: Content-difference threshold for a cut. PySceneDetect's default is 27;
#: 27 is kept deliberately -- it is well calibrated for ordinary footage, and
#: lowering it produces spurious cuts on camera shake and exposure shifts.
CONTENT_THRESHOLD = 27.0

#: Ignore cuts closer together than this. Below roughly half a second a "scene"
#: is a flash or a shake, not something an editor would ever cut on.
MIN_SCENE_LEN_FRAMES = 15

#: Decoding every frame of a long video costs minutes. Analysing every other
#: frame halves that for no meaningful loss of boundary accuracy at 24-60 fps.
FRAME_SKIP = 1


class SceneAnalyzer:
    """Shot boundaries for video."""

    name = AnalyzerName.SCENES
    version = SCENES_ANALYZER_VERSION
    kind = AnalyzerKind.CPU

    def __init__(
        self,
        threshold: float = CONTENT_THRESHOLD,
        min_scene_len: int = MIN_SCENE_LEN_FRAMES,
        frame_skip: int = FRAME_SKIP,
    ) -> None:
        self._threshold = threshold
        self._min_scene_len = min_scene_len
        self._frame_skip = frame_skip

    def supports(self, media_kind: MediaKind) -> bool:
        return media_kind is MediaKind.VIDEO

    def analyze(self, source: AnalysisSource) -> AnalysisOutcome:
        if not self.supports(source.kind):
            return AnalysisOutcome(
                analyzer=self.name,
                version=self.version,
                status=AnalysisStatus.UNSUPPORTED,
                payload={"reason": f"scene detection does not apply to {source.kind.value}"},
            )

        try:
            video = open_video(source.local_path)
        except Exception as exc:
            raise UnsupportedMediaError("PySceneDetect could not open this video") from exc

        manager = SceneManager()
        manager.add_detector(
            ContentDetector(threshold=self._threshold, min_scene_len=self._min_scene_len)
        )
        manager.detect_scenes(video, frame_skip=self._frame_skip, show_progress=False)
        scene_list = manager.get_scene_list()

        scenes = [
            {
                "scene_id": index,
                "start_ms": int(start.get_seconds() * 1000),
                "end_ms": int(end.get_seconds() * 1000),
                "duration_ms": int((end.get_seconds() - start.get_seconds()) * 1000),
                "start_frame": start.get_frames(),
                "end_frame": end.get_frames(),
            }
            for index, (start, end) in enumerate(scene_list)
        ]

        # An empty list means no cuts were found, which for a single continuous
        # take is the correct answer, not a failure. Report it as one scene
        # spanning the clip so downstream code never special-cases "no scenes".
        if not scenes and source.duration_ms:
            scenes = [
                {
                    "scene_id": 0,
                    "start_ms": 0,
                    "end_ms": source.duration_ms,
                    "duration_ms": source.duration_ms,
                    "start_frame": 0,
                    "end_frame": -1,
                }
            ]

        durations = [s["duration_ms"] for s in scenes]
        return AnalysisOutcome(
            analyzer=self.name,
            version=self.version,
            status=AnalysisStatus.OK,
            payload={
                "scenes": scenes,
                "scene_count": len(scenes),
                "mean_scene_ms": int(sum(durations) / len(durations)) if durations else 0,
                "shortest_scene_ms": min(durations) if durations else 0,
                "longest_scene_ms": max(durations) if durations else 0,
                "detector": "content",
                "threshold": self._threshold,
                "min_scene_len_frames": self._min_scene_len,
                "frame_skip": self._frame_skip,
                "used_proxy": source.used_proxy,
            },
        )
