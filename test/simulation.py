import unittest
import src.simulation as sim
import numpy as np


class RecordKeyframes(unittest.TestCase):
    def setUp(self) -> None:
        self.orientation1 = np.array([-9.50095535, 4.10391213, -3.19310876, 3.7417457])
        self.center1 = np.array([6.97865048, 1.03853695, -3.51660027])
        self.orientation2 = np.array([9.04432244, -1.32326678, -7.38508187, -8.32876612])
        self.center2 = np.array([5.77139433, -3.05585285, 3.89290385])
        self.orientation_keyframes = np.array([[np.nan, np.nan, np.nan, np.nan],
                                               [np.nan, np.nan, np.nan, np.nan]],
                                              dtype=np.float64)

        self.center_keyframes = np.array([[np.nan, np.nan, np.nan],
                                          [np.nan, np.nan, np.nan]],
                                         dtype=np.float64)

    def test_first_keyframe(self) -> None:
        orientation_keyframes = np.array([[np.nan, np.nan, np.nan, np.nan],
                                          [np.nan, np.nan, np.nan, np.nan]],
                                         dtype=np.float64)
        center_keyframes = np.array([[np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan]],
                                    dtype=np.float64)

        sim.record_keyframes(self.orientation1,
                             self.center1,
                             orientation_keyframes[0],
                             center_keyframes[0])
        expected_orientation_keyframes = np.array([[-9.50095535, 4.10391213, -3.19310876, 3.7417457],
                                                   [np.nan, np.nan, np.nan, np.nan]],
                                                  dtype=np.float64)
        expected_center_keyframes = np.array([[6.97865048, 1.03853695, -3.51660027],
                                              [np.nan, np.nan, np.nan]],
                                             dtype=np.float64)

        orientation_comparison = np.equal(orientation_keyframes, expected_orientation_keyframes)
        orientation_comparison[np.isnan(orientation_keyframes)
                               & np.isnan(expected_orientation_keyframes)] = True
        center_comparison = np.equal(center_keyframes, expected_center_keyframes)
        center_comparison[np.isnan(center_keyframes)
                          & np.isnan(center_keyframes)] = True

        self.assertTrue(np.all(orientation_comparison))
        self.assertTrue(np.all(center_comparison))

    def test_mutation_of_original_orientation(self) -> None:
        orientation_keyframes = np.array([[np.nan, np.nan, np.nan, np.nan],
                                          [np.nan, np.nan, np.nan, np.nan]],
                                         dtype=np.float64)
        center_keyframes = np.array([[np.nan, np.nan, np.nan],
                                     [np.nan, np.nan, np.nan]],
                                    dtype=np.float64)

        sim.record_keyframes(self.orientation2,
                             self.center2,
                             orientation_keyframes[1],
                             center_keyframes[1])
        orientation_keyframes[1, 0] = 999.

        expected_orientation_keyframes = np.array([[np.nan, np.nan, np.nan, np.nan],
                                                   [999., -1.32326678, -7.38508187, -8.32876612]],
                                                  dtype=np.float64)

        orientation_comparison = np.equal(orientation_keyframes, expected_orientation_keyframes)
        orientation_comparison[np.isnan(orientation_keyframes)
                               & np.isnan(expected_orientation_keyframes)] = True

        self.assertTrue(np.all(orientation_comparison))

        expected_original_orientation = np.array([9.04432244, -1.32326678, -7.38508187, -8.32876612])
        self.assertTrue(np.array_equal(self.orientation2, expected_original_orientation))  # add assertion here


if __name__ == '__main__':
    unittest.main()
