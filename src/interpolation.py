import numpy as np


def linear_interpolation(keyframes: np.ndarray,
                         num: int) -> np.ndarray:
    """
    Linear interpolation.
    :param keyframes: 2 x k numpy array. Generate values between keyframes[0, i] and keyframes[1, i].
    :param num: number of values generated between two keyframes (inclusive).
    :return: k-numpy-array of generated values.
    """
    assert keyframes.dtype == np.float64
    assert len(keyframes.shape) == 2 and keyframes.shape[0] == 2
    assert 0 <= num

    return (np.linspace([0, 1],
                        [1, 1],
                        num=num)
            @ np.array([[-1, 1],
                        [1, 0]])
            @ keyframes)
