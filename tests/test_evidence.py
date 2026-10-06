"""Offline tests for the released evidence builder: pure-python/numpy only.

Evidence-construction tests: keyframe selection,
badge/mark rendering and proposal ranking. The package's own training, grammar
and reward machinery belongs to an earlier line of work that this method does
not use, so it is neither shipped nor tested here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # tests stay runnable via the plain-python fallback runner
    pytest = None

from data.evidence import (badge_keyframe, classical_proposals, draw_pointer_marks,
                                        energy_keyframes, keyframe_change_pair, rank_proposals,
                                        select_keyframes, stratified_keyframes)
from data.evidence_utils import SLOTS, structured_answer

GRAMMAR = {
    "slot_vocab": {
        "status": ["active", "null"],
        "verb": ["insert", "screw", "null"],
        "manipulated_object": ["hex screw", "nut", "not applicable"],
        "target_object": ["hole c2", "stud", "not applicable"],
        "tool": ["hex screwdriver", "no tool"],
    },
    "tuples": [["screw", "hex screw", "hole c2", "hex screwdriver"]],
    "class_counts": {"verb": {"insert": 10, "screw": 1000, "null": 100}},
}

REWARD_CFG = {"grpo": {"reward": {"w_format": 0.5, "w_slot": 2.0, "w_valid": 0.5, "w_composite": 2.0,
                                  "macro_weighted": True, "validity_reward": True,
                                  "macro_alpha": 0.5, "max_class_weight": 10.0}}}

def test_stratified_keyframes_cover_time():
    scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=np.float32)
    picks = stratified_keyframes(scores, 4)
    assert picks == sorted(picks)
    assert len(picks) == 4
    assert picks[0] < 2 and picks[-1] >= 6


def test_energy_keyframes_concentrate_on_action():
    scores = np.zeros(40, dtype=np.float32)
    scores[20:30] = 10.0                      # single action burst
    picks = energy_keyframes(scores, 6)
    assert len(picks) == 6 and picks == sorted(picks)
    in_burst = sum(1 for p in picks if 18 <= p <= 31)
    assert in_burst >= 4                      # budget concentrates on the burst
    assert min(picks) < 18                    # uniform floor keeps global coverage


def test_energy_keyframes_static_clip_falls_back_to_uniform():
    picks = energy_keyframes(np.zeros(30, dtype=np.float32), 5)
    assert len(picks) == 5
    gaps = np.diff(picks)
    assert gaps.max() - gaps.min() <= 3       # roughly even spacing


def test_select_keyframes_methods():
    scores = np.asarray([0.0, 1.0, 0.0, 5.0, 0.0, 1.0], dtype=np.float32)
    assert select_keyframes(scores, 3, "energy") == energy_keyframes(scores, 3)
    assert select_keyframes(scores, 3, "stratified") == stratified_keyframes(scores, 3)
    assert len(select_keyframes(scores, 3, "uniform")) == 3
    assert select_keyframes(scores, 2, "maxmotion") == [1, 3]
    ends = select_keyframes(scores, 3, "ends_uniform")
    assert ends[0] == 0 and ends[-1] == len(scores) - 1
    try:
        select_keyframes(scores, 3, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_keyframe_change_pair_pixel_arithmetic():
    frames = [np.full((60, 80, 3), value, dtype=np.uint8) for value in (10, 12, 200, 202)]
    assert keyframe_change_pair(frames, [0, 1, 2, 3]) == 2


def test_badge_and_marks_render():
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3)).astype(np.uint8)
    badged = badge_keyframe(frame, 2, 6, 0.25, 224)
    assert badged.shape[1] == 224 and badged.ndim == 3
    marked, marks = draw_pointer_marks(frame, [(10, 10, 60, 60), (100, 80, 180, 160)], 224)
    assert len(marks) == 2 and marks[0]["id"] == 1
    assert marked.shape[1] == 224
    # badge and marks must actually change pixels
    assert not np.array_equal(badged[:20, :40], frame[:20, :40])


def test_rank_proposals_prefers_hand_adjacent_moving_regions():
    motion = np.zeros((200, 300), dtype=np.float32)
    motion[80:120, 40:90] = 20.0     # moving region near the hand
    motion[20:60, 220:280] = 20.0    # moving region far from the hand (occlusion flicker)
    hand = [np.asarray([70.0, 100.0], dtype=np.float32)]
    near, far, static_far = (40, 80, 90, 120), (220, 20, 280, 60), (200, 150, 260, 190)
    kept = rank_proposals([far, static_far, near], motion, hand, 3, hand_scale_px=50.0,
                          min_regions=1)
    assert kept[0] == near
    # the adaptive cutoff drops far/static leftovers instead of padding to max
    assert static_far not in kept
    # without a hand track, motion is the only signal: both moving regions rank first
    kept_no_hand = rank_proposals([static_far, far, near], motion, [], 3)
    assert set(kept_no_hand[:2]) == {near, far}


def test_classical_proposals_return_boxes():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[50:120, 60:140] = 255
    frame[150:200, 200:280] = 180
    boxes = classical_proposals(frame, 8, 0.0008)
    assert boxes, "expected at least one proposal"
    assert all(len(box) == 4 for box in boxes)


