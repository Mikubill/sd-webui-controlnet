import unittest
import numpy as np

import importlib
utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()

from annotator.openpose.util import faceDetect, handDetect

class TestFaceDetect(unittest.TestCase):
    def test_no_faces(self):
        candidate = np.array([])
        subset = np.array([])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = []
        result = faceDetect(candidate, subset, oriImg)

        self.assertEqual(result, expected_result)

    def test_single_face(self):
        candidate = np.array([
            [50, 50], [30, 40], [70, 40], [20, 50], [80, 50]
        ])
        subset = np.array([
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, 2, 0.5]
        ])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = [[0, 0, 120]]
        result = faceDetect(candidate, subset, oriImg)

        self.assertEqual(result, expected_result)

    def test_multiple_faces(self):
        candidate = np.array([
            [50, 50], [30, 40], [70, 40], [20, 50], [80, 50],
            [25, 25], [10, 20], [40, 20], [5, 25], [45, 25]
        ])
        subset = np.array([
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 2, 3, 4, 2, 0.5],
            [5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, 8, 9, 2, 0.5]
        ])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = [[0, 0, 120], [0, 0, 90]]
        result = faceDetect(candidate, subset, oriImg)

        self.assertEqual(result, expected_result)


class TestHandDetect(unittest.TestCase):
    def test_no_hands(self):
        candidate = np.array([])
        subset = np.array([])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = []
        result = handDetect(candidate, subset, oriImg)

        self.assertEqual(result, expected_result)

    def test_single_left_hand(self):
        candidate = np.array([
            [20, 20], [40, 30], [60, 40], [20, 60], [40, 70], [60, 80]
        ])
        subset = np.array([
            [-1, -1, -1, -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0.5]
        ])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = [[49, 26, 33, True]]
        result = handDetect(candidate, subset, oriImg)

        self.assertEqual(result, expected_result)

    def test_single_right_hand(self):
        candidate = np.array([
            [20, 20], [40, 30], [60, 40], [20, 60], [40, 70], [60, 80]
        ])
        subset = np.array([
            [-1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0.5]
        ])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = [[49, 26, 33, False]]
        result = handDetect(candidate, subset, oriImg)

        self.assertEqual(result, expected_result)

    def test_multiple_hands(self):
        candidate = np.array([
            [20, 20], [40, 30], [60, 40], [20, 60], [40, 70], [60, 80],
            [10, 10], [30, 20], [50, 30], [10, 50], [30, 60], [50, 70]
        ])
        subset = np.array([
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0.5],
            [6, 7, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 0.5]
        ])
        oriImg = np.zeros((100, 100, 3), dtype=np.uint8)

        expected_result = [[16, 43, 56, False], [6, 33, 60, False]]
        result = handDetect(candidate, subset, oriImg)
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
