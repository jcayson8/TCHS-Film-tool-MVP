"""Regression tests for classical tracking frame order and failure handling."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import tracker as tracker_service  # noqa: E402
from app.services.tracker import track  # noqa: E402


class ScriptedTracker:
    """Small deterministic stand-in for CSRT's init/update interface."""

    def __init__(self, boxes: list[tuple[float, float, float, float]]) -> None:
        self.boxes = iter(boxes)

    def init(self, _image: np.ndarray, _box: tuple[int, int, int, int]) -> bool:
        return True

    def update(self, _image: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        return True, next(self.boxes)


class TrackingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generator = np.random.default_rng(17)
        cls.player_one = generator.integers(0, 256, (70, 38, 3), dtype=np.uint8)
        cls.player_two = generator.integers(0, 256, (58, 34, 3), dtype=np.uint8)

    def frame(self, index: int, include_players: bool = True) -> np.ndarray:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        if include_players:
            image[60:130, 45 + index * 4:83 + index * 4] = self.player_one
            image[145:203, 220 - index * 3:254 - index * 3] = self.player_two
        return image

    @staticmethod
    def payload(image: np.ndarray) -> tuple[bytes, str]:
        encoded, buffer = cv2.imencode(".png", image)
        if not encoded:
            raise AssertionError("Synthetic frame encoding failed")
        return buffer.tobytes(), "image/png"

    @staticmethod
    def boxes(count: int = 1) -> str:
        values = [{
            "player_id": "player-one",
            "class_index": 2,
            "x": 45 / 320,
            "y": 60 / 240,
            "width": 38 / 320,
            "height": 70 / 240,
        }]
        if count == 2:
            values.append({
                "player_id": "player-two",
                "class_index": 5,
                "x": 220 / 320,
                "y": 145 / 240,
                "width": 34 / 320,
                "height": 58 / 240,
            })
        return json.dumps(values)

    def scripted_track(
        self,
        raw_boxes: list[tuple[float, float, float, float]],
        confidence: list[tuple[str, float]] | None = None,
    ) -> dict:
        initial = self.payload(self.frame(0))
        payloads = [self.payload(self.frame(0)) for _ in raw_boxes]
        factory = lambda: ScriptedTracker(raw_boxes)  # noqa: E731
        confidence_values = confidence or [("high", 1.0)] * len(raw_boxes)
        with (
            patch.object(tracker_service, "_tracker_factory", return_value=("CSRT", factory)),
            patch.object(tracker_service, "_confidence", side_effect=confidence_values),
        ):
            return track(
                initial[0],
                initial[1],
                payloads,
                self.boxes(),
                [6900 + index * 33 for index in range(len(raw_boxes))],
                [207 + index for index in range(len(raw_boxes))],
            )

    def test_single_player_tracks_sequential_absolute_frames(self) -> None:
        payloads = [self.payload(self.frame(index)) for index in range(6)]
        result = track(
            payloads[0][0],
            payloads[0][1],
            payloads[1:],
            self.boxes(),
            [6900, 6933, 6967, 7000, 7033],
            [207, 208, 209, 210, 211],
        )
        self.assertEqual(result["tracker"], "CSRT")
        self.assertEqual([frame["video_frame_number"] for frame in result["frames"]], [207, 208, 209, 210, 211])
        self.assertFalse(result["failures"])
        self.assertGreater(
            len({round(frame["annotations"][0]["x"], 5) for frame in result["frames"]}),
            1,
        )

    def test_multiple_players_preserve_identity_and_class(self) -> None:
        payloads = [self.payload(self.frame(index)) for index in range(4)]
        result = track(
            payloads[0][0],
            payloads[0][1],
            payloads[1:],
            self.boxes(2),
            [6900, 6933, 6967],
            [207, 208, 209],
        )
        for frame in result["frames"]:
            annotations = {item["player_id"]: item for item in frame["annotations"]}
            self.assertEqual(set(annotations), {"player-one", "player-two"})
            self.assertEqual(annotations["player-one"]["class_index"], 2)
            self.assertEqual(annotations["player-two"]["class_index"], 5)

    def test_partial_success_is_preserved_at_exact_failure_frame(self) -> None:
        initial = self.payload(self.frame(0))
        first = self.payload(self.frame(1))
        blank = self.payload(self.frame(2, include_players=False))
        result = track(
            initial[0],
            initial[1],
            [first, blank],
            self.boxes(),
            [6900, 6933],
            [207, 208],
        )
        self.assertEqual(len(result["frames"][0]["annotations"]), 1)
        self.assertEqual(result["frames"][1]["annotations"], [])
        self.assertEqual(result["failures"][0]["video_frame_number"], 208)
        self.assertEqual(result["failures"][0]["reason"], "Tracker lost the player.")

    def test_skipped_frame_numbers_are_rejected(self) -> None:
        payloads = [self.payload(self.frame(index)) for index in range(3)]
        with self.assertRaisesRegex(HTTPException, "sequential and ordered"):
            track(
                payloads[0][0],
                payloads[0][1],
                payloads[1:],
                self.boxes(),
                [6900, 6933],
                [207, 209],
            )

    def test_stable_fifteen_frame_track_is_accepted(self) -> None:
        raw_boxes = [(45 + index * 2, 60, 38, 70) for index in range(1, 16)]
        result = self.scripted_track(raw_boxes)
        self.assertEqual(result["completed_frame_count"], 15)
        self.assertFalse(result["failures"])
        self.assertTrue(all(frame["annotations"] for frame in result["frames"]))

    def test_gradual_movement_and_scale_change_remain_accepted(self) -> None:
        raw_boxes = [
            (45 + index * 3, 60 - index * 0.25, 38 * (1 + index * 0.015), 70 * (1 + index * 0.015))
            for index in range(1, 21)
        ]
        result = self.scripted_track(raw_boxes)
        self.assertEqual(result["completed_frame_count"], 20)
        self.assertFalse(result["failures"])

    def test_sudden_implausible_jump_stops_at_exact_frame(self) -> None:
        result = self.scripted_track([
            (49, 60, 38, 70),
            (280, 160, 38, 70),
            (284, 160, 36, 70),
        ])
        self.assertEqual(result["failures"][0]["video_frame_number"], 208)
        self.assertEqual(len(result["frames"][0]["annotations"]), 1)
        self.assertEqual(result["frames"][1]["annotations"], [])
        self.assertEqual(result["completed_frame_count"], 2)

    def test_sudden_large_size_change_stops(self) -> None:
        result = self.scripted_track([
            (49, 60, 38, 70),
            (49, 40, 100, 180),
        ])
        self.assertEqual(result["failures"][0]["video_frame_number"], 208)
        self.assertEqual(result["frames"][1]["annotations"], [])

    def test_two_consecutive_low_appearance_frames_stop(self) -> None:
        result = self.scripted_track(
            [(49, 60, 38, 70), (53, 60, 38, 70), (57, 60, 38, 70)],
            [("low", 0.2), ("low", 0.2)],
        )
        self.assertEqual(result["frames"][0]["annotations"][0]["tracking_confidence"], "low")
        self.assertEqual(result["frames"][1]["annotations"], [])
        self.assertEqual(result["failures"][0]["video_frame_number"], 208)
        self.assertEqual(result["completed_frame_count"], 2)

    def test_isolated_weak_frame_followed_by_recovery(self) -> None:
        result = self.scripted_track(
            [(49, 60, 38, 70), (53, 60, 38, 70), (57, 60, 38, 70)],
            [("low", 0.2), ("high", 0.9), ("high", 0.9)],
        )
        self.assertFalse(result["failures"])
        self.assertEqual(
            [frame["annotations"][0]["tracking_confidence"] for frame in result["frames"]],
            ["low", "high", "high"],
        )

    def test_weak_drift_does_not_advance_trusted_geometry(self) -> None:
        result = self.scripted_track(
            [
                (250, 60, 38, 70),
                (270, 165, 38, 70),
            ],
            [("low", 0.2)],
        )
        first = result["frames"][0]["annotations"]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["tracking_confidence"], "low")
        self.assertEqual(result["frames"][1]["annotations"], [])
        self.assertEqual(result["failures"][0]["video_frame_number"], 208)
        self.assertEqual(result["completed_frame_count"], 2)

    def test_output_boxes_remain_inside_normalized_bounds(self) -> None:
        result = self.scripted_track([
            (20, 40, 50, 90),
            (5, 25, 55, 95),
            (-10, 10, 60, 100),
        ])
        self.assertFalse(result["failures"])
        for frame in result["frames"]:
            box = frame["annotations"][0]
            self.assertGreaterEqual(box["x"], 0)
            self.assertGreaterEqual(box["y"], 0)
            self.assertGreater(box["width"], 0)
            self.assertGreater(box["height"], 0)
            self.assertLessEqual(box["x"] + box["width"], 1)
            self.assertLessEqual(box["y"] + box["height"], 1)

    def test_repeated_severe_boundary_clipping_stops_after_tolerance(self) -> None:
        result = self.scripted_track([
            (-24, 60, 60, 70),
            (-24, 60, 60, 70),
        ])
        self.assertEqual(len(result["frames"][0]["annotations"]), 1)
        self.assertEqual(result["frames"][1]["annotations"], [])
        self.assertEqual(result["failures"][0]["video_frame_number"], 208)


if __name__ == "__main__":
    unittest.main()
