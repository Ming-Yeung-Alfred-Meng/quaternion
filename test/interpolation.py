import unittest
import numpy as np
import src.interpolation as i


class LinearInterpolation(unittest.TestCase):
    def test_something(self):
        keyframes = np.array([[0.92387953, 0.38268343, 0., 0.],
                              [0.92387953, -0.38268343, -0., -0.]])
        expected = np.array([[0.92387953, 0.38268343, 0., 0.],
                             [0.92387953, 0.29764267, 0., 0.],
                             [0.92387953, 0.21260191, 0., 0.],
                             [0.92387953, 0.12756114, 0., 0.],
                             [0.92387953, 0.04252038, 0., 0.],
                             [0.92387953, -0.04252038, 0., 0.],
                             [0.92387953, -0.12756114, 0., 0.],
                             [0.92387953, -0.21260191, 0., 0.],
                             [0.92387953, -0.29764267, 0., 0.],
                             [0.92387953, -0.38268343, 0., 0.]])
        actual = i.linear_interpolation(keyframes, 10)

        self.assertTrue(np.allclose(actual, expected))


if __name__ == '__main__':
    unittest.main()
