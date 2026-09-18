"""The brief: what the model is allowed to know about the footage.

Two jobs, and the second one matters more than the first.

**Summarise.** Turn analysis rows into a compact table a model can reason over:
duration, shape, sharpness, exposure, contrast, scene count, whether anyone is
on screen, whether there is audio. Numbers, normalised, rounded -- not raw
analyzer payloads with their per-frame arrays and model version strings.

**Namespace.** Every clip is given a short handle, ``c1``, ``c2``, ``c3``. The
model sees handles and emits handles. It never sees a UUID, a filename, a
storage key or a path, and it has no vocabulary in which to name a file that is
not in the brief.

That second point is the security design of the whole feature. A model cannot
reference another project's media because that media has no handle; it cannot
invent a reference because ``c99`` resolves to nothing and is rejected; it
cannot leak a filename in the output because it was never told one.

Filenames are withheld for a second reason as well. A file called
``ignore-previous-instructions-and-....mp4`` is a prompt injection that a user
can plant simply by naming a file, and a media library is exactly the kind of
place where names arrive from outside. Excluding them costs the model a little
context about intent, and removes the entire class of attack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visionforge.domain.editplan import MAX_STILL_MS
from visionforge.domain.ids import MediaId
from visionforge.domain.media import MediaKind
from visionforge.domain.selection import ScoredCandidate, SelectionResult

#: Ceiling on how many clips are described to the model. Beyond this the brief
#: stops being a summary and starts being a cost: the strongest candidates are
#: kept and the rest are reported as a count, so the model knows the library was
#: larger than what it can see.
MAX_BRIEF_CLIPS = 40

#: Ceiling on the user's own request text. Long enough for a real description,
#: short enough that nobody can paste a document into the prompt.
MAX_REQUEST_CHARS = 500


@dataclass(frozen=True, slots=True)
class ClipBrief:
    """One clip, as the model sees it.

    Everything here is a number, a bounded enum, or a boolean. There is no field
    a filename, a path or an instruction could travel in -- which means the
    brief cannot carry an injection even if the analysis rows somehow did.
    """

    #: ``c1``, ``c2``, ... The model's entire vocabulary for referring to media.
    handle: str
    duration_ms: int
    width: int | None
    height: int | None
    #: 0..1, the selection score. The one number that already blends the others.
    score: float
    #: 0..1 each, so the model compares clips rather than guessing at scales.
    sharpness: float | None
    exposure: float | None
    contrast: float | None
    scene_count: int | None
    has_faces: bool | None
    has_audio: bool
    #: Position in the upload order, 1-based. The only chronology available.
    sequence: int
    #: A photo rather than a video (Phase 12). Its ``duration_ms`` is then the
    #: longest it may be held, not a length it has.
    still: bool = False

    def as_payload(self) -> dict[str, Any]:
        """Compact JSON for the prompt. Omits unknowns rather than sending null.

        A model reading ``"sharpness": null`` tends to reason about the null.
        A model that simply does not see the key treats it as unknown, which is
        what it is.
        """
        payload: dict[str, Any] = {
            "id": self.handle,
            "seq": self.sequence,
            "duration_ms": self.duration_ms,
            "score": round(self.score, 3),
        }
        if self.width and self.height:
            payload["size"] = f"{self.width}x{self.height}"
        for key, value in (
            ("sharpness", self.sharpness),
            ("exposure", self.exposure),
            ("contrast", self.contrast),
        ):
            if value is not None:
                payload[key] = round(value, 2)
        if self.scene_count is not None:
            payload["scenes"] = self.scene_count
        if self.has_faces is not None:
            payload["faces"] = self.has_faces
        payload["audio"] = self.has_audio
        if self.still:
            payload["still"] = True
        return payload


@dataclass(frozen=True, slots=True)
class EditBrief:
    """The complete input to a planning model.

    ``handles`` is the resolution table, and it stays on this side of the
    boundary. The model receives ``clips``; the planner keeps the mapping and
    uses it to turn the model's answer back into media ids.
    """

    clips: tuple[ClipBrief, ...]
    handles: dict[str, MediaId]
    #: How many usable clips existed before the brief was truncated.
    considered: int
    #: How many were rejected by the usability gates, and why, so the model is
    #: not told to "use everything" when a third of the library is unusable.
    rejected: int
    total_source_ms: int

    def media_id_for(self, handle: str) -> MediaId | None:
        return self.handles.get(handle)

    def as_payload(self) -> dict[str, Any]:
        return {
            "clips": [clip.as_payload() for clip in self.clips],
            "clip_count": len(self.clips),
            "considered": self.considered,
            "rejected_unusable": self.rejected,
            "total_source_ms": self.total_source_ms,
        }


def _normalised_sharpness(scored: ScoredCandidate) -> float | None:
    return _component(scored, "sharpness")


def _component(scored: ScoredCandidate, key: str) -> float | None:
    value = scored.components.get(key)
    return None if value is None else float(value)


def build_brief(selection: SelectionResult, *, limit: int = MAX_BRIEF_CLIPS) -> EditBrief:
    """Summarise a completed selection for a model.

    Built from the *selection*, not from raw candidates, for two reasons. The
    scores and normalised components already exist there, so nothing is
    recomputed; and clips the deterministic gates rejected as unusable never
    reach the model at all. Asking a model to avoid a black frame is strictly
    worse than not offering it one.

    Handles are assigned in selection order -- strongest first -- so ``c1`` is
    always the best clip by the technical ranking. That is a useful prior for
    the model and a stable one for tests.
    """
    chosen = list(selection.selected)[:limit]

    clips: list[ClipBrief] = []
    handles: dict[str, MediaId] = {}
    for index, scored in enumerate(chosen, start=1):
        handle = f"c{index}"
        candidate = scored.candidate
        handles[handle] = candidate.media_id
        still = candidate.kind is MediaKind.IMAGE
        clips.append(
            ClipBrief(
                handle=handle,
                duration_ms=MAX_STILL_MS if still else candidate.duration_ms or 0,
                width=candidate.width,
                height=candidate.height,
                score=scored.score,
                sharpness=_normalised_sharpness(scored),
                exposure=_component(scored, "exposure"),
                contrast=_component(scored, "contrast"),
                scene_count=candidate.scene_count,
                has_faces=candidate.has_faces,
                has_audio=candidate.has_audio,
                sequence=candidate.sequence + 1,
                still=still,
            )
        )

    return EditBrief(
        clips=tuple(clips),
        handles=handles,
        considered=len(selection.selected),
        rejected=len(selection.rejected),
        total_source_ms=sum(clip.duration_ms for clip in clips),
    )


def clean_request_text(text: str | None) -> str | None:
    """Normalise the user's own words before they enter a prompt.

    Whitespace is collapsed and the result is truncated. Nothing is "sanitised"
    beyond that, and deliberately so: stripping suspicious phrases from user
    text is security theatre that breaks legitimate requests, because the
    defence that actually holds is downstream -- the model's output is parsed
    into a closed vocabulary of handles and integers, so the worst a successful
    injection achieves is a differently-paced edit of the user's own footage.

    What this *does* prevent is the request smuggling structure into the prompt:
    it is delivered inside a delimited block, and its length is bounded.
    """
    if text is None:
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    return collapsed[:MAX_REQUEST_CHARS]


__all__ = [
    "MAX_BRIEF_CLIPS",
    "MAX_REQUEST_CHARS",
    "ClipBrief",
    "EditBrief",
    "build_brief",
    "clean_request_text",
]
