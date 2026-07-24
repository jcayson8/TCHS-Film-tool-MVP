"""Regression tests for classical tracking frame order and failure handling."""

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.tracker import track  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
