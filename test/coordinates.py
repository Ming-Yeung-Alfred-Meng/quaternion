import unittest
import numpy as np
import src.coordinates as c


class RecordKeyframe(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = np.array([9., -10., 6.])

    def test_one_vector(self) -> None:
        vector = np.array([-10., -5., -4.])
        actual = np.array([np.nan, np.nan, np.nan])
        c.local_coordinates(actual, vector, self.origin)
        expected = np.array([-19., 5., -10.])
        self.assertTrue(np.allclose(actual, expected))

    def test_multiple_vectors(self) -> None:
        vectors = np.array([[1., -8., 1.],
                            [-10., -1., -8.],
                            [5., 2., 3.]])

        actual = np.array([[np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.nan]])
        c.local_coordinates(actual, vectors, self.origin)
        expected = np.array([[-8., 2., -5.],
                             [-19., 9., -14.],
                             [-4., 12., -3.]])

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
