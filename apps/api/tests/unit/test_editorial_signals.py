"""Editorial signals, the signal board, and the event vocabulary.

The claim under test throughout is the one that makes the rest of the phase
defensible: **an observation is only made when the evidence supports it.** A
clip nobody measured gets ``UNKNOWN``, not a plausible guess; a weak claim is
reported with a low confidence rather than as a fact; and a signal whose
analyzer never ran is ``None`` rather than zero.
"""

from __future__ import annotations

from tests.unit.editorial_fixtures import clip, football_project, vector
from visionforge.domain.editorial import (
    MAX_EVENTS_PER_CLIP,
    MIN_EVENT_CONFIDENCE,
    ClipReading,
    DetectedEvent,
    EditorialEvent,
    EventContext,
    board_from,
    cosine,
    detect_events,
    read_board,
    signals_for,
)
from visionforge.domain.selection import score_candidate


# ------------------------------------------------------------------- signals
class TestSignals:
    def test_an_unmeasured_signal_is_none_not_zero(self) -> None:
        """The distinction the whole layer rests on.

        Zero motion means "measured, and nothing moved". ``None`` means nobody
        looked. Collapsing the two would let an unanalysed clip be reported as
        an establishing shot.
        """
        signals = signals_for(clip(0, motion=None, motion_spread=None, saturation=None))
        assert signals.motion is None
        assert signals.motion_variation is None
        assert signals.saturation is None

    def test_motion_normalises_against_a_fixed_reference(self) -> None:
        """Not against the batch: a clip's signals must not depend on its peers."""
        alone = signals_for(clip(0, motion=0.175))
        crowded = signals_for(clip(0, motion=0.175))
        assert alone.motion == crowded.motion == 0.5

    def test_motion_saturates_rather_than_exceeding_one(self) -> None:
        assert signals_for(clip(0, motion=0.9)).motion == 1.0

    def test_energy_is_total_even_with_nothing_measured(self) -> None:
        """A clip with no dynamics row still has an energy, and it is 0.0.

        Total rather than optional, so no caller has to special-case it -- the
        one place a default is allowed in this module, and it is stated.
        """
        assert signals_for(clip(0, motion=None, motion_spread=None, saturation=None)).energy == 0.0

    def test_energy_is_led_by_motion(self) -> None:
        calm = signals_for(clip(0, motion=0.02, saturation=0.9))
        busy = signals_for(clip(0, motion=0.34, saturation=0.1))
        assert busy.energy > calm.energy

    def test_quality_is_carried_from_the_ranker_not_recomputed(self) -> None:
        """One notion of "sharp", shared with selection.

        Recomputing it here would produce a second normalisation that drifts the
        next time a reference constant is tuned.
        """
        candidate = clip(0)
        scored = score_candidate(candidate)
        assert signals_for(candidate, scored=scored).quality == round(scored.score, 4)

    def test_face_presence_is_the_share_of_sampled_frames(self) -> None:
        signals = signals_for(clip(0, faces=(False, False, True, True, True)))
        assert signals.face_presence == 0.6

    def test_face_presence_falls_back_to_the_boolean_when_frames_are_absent(self) -> None:
        """An older analysis row still supports the coarsest honest claim."""
        candidate = clip(0, faces=())
        assert signals_for(candidate).face_presence == 0.0

    def test_the_payload_omits_unmeasured_signals(self) -> None:
        """A reader seeing no key treats it as unknown, which is what it is."""
        payload = signals_for(
            clip(0, motion=None, motion_spread=None, saturation=None)
        ).as_payload()
        assert "motion" not in payload
        assert "saturation" not in payload
        assert payload["energy"] == 0.0


# --------------------------------------------------------------------- board
class TestSignalBoard:
    def test_rank_is_a_percentile_within_the_project(self) -> None:
        """ "High motion" is not absolute.

        A nature project's busiest shot and a football project's busiest shot
        are nothing alike, and one fixed threshold would find eleven peaks in
        one and none in the other.
        """
        board = board_from([clip(i, motion=0.05 * (i + 1)) for i in range(5)])
        ranks = [board.rank(signal.media_id, "motion") for signal in board.signals]
        assert ranks == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_rank_is_none_with_too_few_measurements(self) -> None:
        board = board_from([clip(0, motion=0.2)])
        assert board.rank(board.signals[0].media_id, "motion") is None

    def test_ties_rank_at_the_midpoint_of_their_band(self) -> None:
        board = board_from([clip(i, motion=0.2) for i in range(3)])
        assert {board.rank(s.media_id, "motion") for s in board.signals} == {0.5}

    def test_semantic_groups_find_the_same_subject(self) -> None:
        clips = [clip(0, subject=1), clip(1, subject=1), clip(2, subject=2)]
        groups = board_from(clips).semantic_groups()
        assert len(groups) == 1
        assert set(groups[0]) == {clips[0].media_id, clips[1].media_id}

    def test_a_clip_with_no_embedding_is_never_grouped(self) -> None:
        """Absence of a vector is not evidence of difference.

        Treating it as unique is exactly the bug that puts five un-embedded
        copies of one shot in an edit.
        """
        clips = [clip(0, subject=1), clip(1, subject=1, unembedded=True)]
        assert board_from(clips).semantic_groups() == ()

    def test_distinctiveness_is_one_against_nothing(self) -> None:
        board = board_from([clip(0)])
        assert board.distinctiveness(board.signals[0].media_id, ()) == 1.0

    def test_distinctiveness_falls_for_a_repeat(self) -> None:
        clips = [clip(0, subject=1), clip(1, subject=1), clip(2, subject=2)]
        board = board_from(clips)
        repeat = board.distinctiveness(clips[1].media_id, (clips[0].media_id,))
        fresh = board.distinctiveness(clips[2].media_id, (clips[0].media_id,))
        assert repeat < fresh

    def test_cosine_rejects_a_missing_or_mismatched_vector(self) -> None:
        assert cosine(None, vector(1)) is None
        assert cosine(vector(1), ()) is None
        assert cosine((0.0, 0.0), (1.0, 0.0)) is None


# -------------------------------------------------------------------- events
class TestEventDetection:
    def test_a_clip_with_nothing_measured_is_unknown(self) -> None:
        """The correct answer, and a usable one -- not a failure."""
        events = detect_events(
            signals_for(clip(0, motion=None, motion_spread=None, saturation=None))
        )
        assert [e.event for e in events] == [EditorialEvent.UNKNOWN]

    def test_establishing_needs_motion_to_have_been_measured(self) -> None:
        """Unexamined is not calm.

        A clip with no dynamics row must not be reported as an establishing
        shot on the strength of the measurement being absent.
        """
        events = detect_events(signals_for(clip(0, motion=None)))
        assert EditorialEvent.ESTABLISHING not in {e.event for e in events}

    def test_a_wide_calm_held_shot_is_establishing(self) -> None:
        events = detect_events(signals_for(clip(0, motion=0.02, duration_ms=9_000)))
        assert EditorialEvent.ESTABLISHING in {e.event for e in events}

    def test_a_large_face_is_a_close_up_and_never_establishing(self) -> None:
        signals = signals_for(clip(0, motion=0.02, faces=(True,) * 5, face_area=0.2))
        found = {e.event for e in detect_events(signals)}
        assert EditorialEvent.CLOSE_UP in found
        assert EditorialEvent.ESTABLISHING not in found

    def test_peak_motion_is_relative_to_the_project(self) -> None:
        signals = signals_for(clip(0, motion=0.30))
        assert EditorialEvent.PEAK_MOTION not in {
            e.event for e in detect_events(signals, EventContext(motion_rank=0.5))
        }
        assert EditorialEvent.PEAK_MOTION in {
            e.event for e in detect_events(signals, EventContext(motion_rank=1.0))
        }

    def test_subject_entry_needs_the_shape_of_the_frame_list(self) -> None:
        entering = signals_for(clip(0, faces=(False, False, True, True, True)))
        leaving = signals_for(clip(0, faces=(True, True, True, False, False)))
        assert EditorialEvent.SUBJECT_ENTRY in {e.event for e in detect_events(entering)}
        assert EditorialEvent.SUBJECT_EXIT in {e.event for e in detect_events(leaving)}

    def test_reaction_needs_a_neighbour(self) -> None:
        """A calm close-up is a portrait; the same shot after a sprint is a
        reaction. Nothing inside the clip distinguishes them."""
        signals = signals_for(clip(0, motion=0.03, faces=(True,) * 5, face_area=0.12))
        alone = {e.event for e in detect_events(signals, EventContext())}
        after = {e.event for e in detect_events(signals, EventContext(previous_energy=0.9))}
        assert EditorialEvent.REACTION not in alone
        assert EditorialEvent.REACTION in after

    def test_dialogue_is_capped_low_because_nothing_listens(self) -> None:
        """The honest half of the weakest claim in the vocabulary."""
        signals = signals_for(clip(0, motion=0.02, faces=(True,) * 5, has_audio=True))
        dialogue = next(e for e in detect_events(signals) if e.event is EditorialEvent.DIALOGUE)
        assert dialogue.confidence <= 0.45
        assert "no_speech_detection" in dialogue.evidence

    def test_dialogue_needs_an_audio_stream(self) -> None:
        signals = signals_for(clip(0, motion=0.02, faces=(True,) * 5, has_audio=False))
        assert EditorialEvent.DIALOGUE not in {e.event for e in detect_events(signals)}

    def test_an_internal_cut_is_a_transition_moment(self) -> None:
        signals = signals_for(clip(0, scene_boundaries_ms=(3_000,)))
        assert EditorialEvent.TRANSITION_MOMENT in {e.event for e in detect_events(signals)}

    def test_ending_is_only_offered_to_the_last_clip(self) -> None:
        """ "This could be an ending" is true of almost any calm shot, and is
        therefore worth nothing unless the clip is actually last."""
        signals = signals_for(clip(0, motion=0.02, luminance=60.0))
        assert EditorialEvent.ENDING not in {
            e.event for e in detect_events(signals, EventContext(is_last=False))
        }
        assert EditorialEvent.ENDING in {
            e.event for e in detect_events(signals, EventContext(is_last=True))
        }

    def test_every_event_clears_the_confidence_floor(self) -> None:
        for candidate in football_project():
            for detected in detect_events(signals_for(candidate)):
                assert detected.confidence >= MIN_EVENT_CONFIDENCE

    def test_a_clip_carries_at_most_three_events(self) -> None:
        """Eight observations about one clip is indistinguishable from none."""
        signals = signals_for(
            clip(
                0,
                motion=0.34,
                motion_spread=0.12,
                faces=(False, True, True, True, True),
                face_area=0.2,
                saturation=0.9,
                scene_boundaries_ms=(2_000, 4_000),
            )
        )
        assert len(detect_events(signals, EventContext(motion_rank=1.0))) <= MAX_EVENTS_PER_CLIP

    def test_events_come_back_strongest_first(self) -> None:
        signals = signals_for(clip(0, motion=0.34, faces=(True,) * 5, face_area=0.25))
        confidences = [e.confidence for e in detect_events(signals, EventContext(motion_rank=1.0))]
        assert confidences == sorted(confidences, reverse=True)

    def test_detection_is_deterministic(self) -> None:
        signals = signals_for(clip(0, motion=0.21, faces=(True, False, True, False, True)))
        assert detect_events(signals) == detect_events(signals)


class TestReadBoard:
    def test_context_uses_upload_order_not_iteration_order(self) -> None:
        """Chronology is the only narrative fact available and is not thrown
        away by whatever order the caller happened to hold the clips in."""
        clips = list(reversed(football_project()))
        readings = read_board(board_from(clips))
        assert [r.signals.sequence for r in readings] == sorted(
            r.signals.sequence for r in readings
        )

    def test_every_clip_gets_at_least_one_reading(self) -> None:
        readings = read_board(board_from(football_project()))
        assert len(readings) == len(football_project())
        assert all(reading.events for reading in readings)

    def test_confidence_for_an_absent_event_is_zero(self) -> None:
        reading = ClipReading(
            signals=signals_for(clip(0)),
            events=(DetectedEvent(EditorialEvent.ACTION, 0.8),),
        )
        assert reading.confidence_for(EditorialEvent.ACTION) == 0.8
        assert reading.confidence_for(EditorialEvent.CELEBRATION) == 0.0
        assert reading.has(EditorialEvent.ACTION)
        assert not reading.has(EditorialEvent.CELEBRATION)
