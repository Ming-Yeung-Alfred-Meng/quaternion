import numpy as np
from typing import Union
from numpy.linalg import norm


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Return the normalized vectors.
    :param vectors: v x d numpy array of d dimensional vectors.
    :return: v x d numpy array of normalized vectors.
    """
    vectors = vectors.copy()

    norms = np.linalg.norm(vectors, axis=-1)
    mask = norms != 0
    vectors[mask] /= norms[mask][None].T

    return vectors


def quaternion_multiplication(q: np.ndarray,
                              p: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions
    :param q: 4-numpy-array representing a quaternion.
    :param p: 4-numpy-array representing a quaternion.
    :return: 4-numpy-array representing the product.
    """
    assert q.dtype == np.float64 and p.dtype == np.float64
    assert q.shape == (4,) and p.shape == (4,)

    result = np.empty(4)
    result[0] = q[0] * p[0] - np.dot(q[1:], p[1:])
    result[1:] = q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])

    return result


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """
    Return the multiplication inverse of a quaternion.
    :param q: 4-numpy-array representing a quaternion.
    :return: 4-numpy-array representing the multiplicative inverse of q.
    """
    assert q.dtype == np.float64
    assert q.shape == (4,)

    result = q / np.linalg.norm(q) ** 2
    result[1:] *= -1

    return result


def rotation_quaternion(angle: Union[float, np.float64, np.ndarray],
                        axis: np.ndarray) -> np.ndarray:
    """
    Return the quaternion representation of an angle-axis rotation.
    :param angle: radian angle to rotate
    :param axis: 3-numpy-array of axis (that crosses the origin) about which to rotate.
    :return: 4-numpy-array quaternion representation of the rotation about the axis by the angle.
    """
    assert axis.dtype == np.float64
    if len(axis.shape) == 1:
        assert type(angle) == float or type(angle) == np.float64
    elif len(axis.shape) == 2:
        assert type(angle) == float or (len(angle.shape) == 1
                                        and angle.shape[0] == axis.shape[0])
    else:
        assert False

    axis = normalize(axis)

    result = np.insert(axis, 0, np.cos(angle / 2), axis=-1)
    result.T[1:] *= -np.sin(angle / 2)

    return result


def vector_to_quaternion(v: np.ndarray) -> np.ndarray:
    """
    Return the quaternion representation of a 3D vector.
    :param v: 3-numpy-array representing a vector.
    :return: quaternion representation of v.
    """
    assert v.dtype == np.float64
    assert len(v.shape) <= 2 and v.shape[-1] == 3

    return np.insert(v, 0, 0.0, axis=-1)


def quaternion_to_vector(q: np.ndarray) -> np.ndarray:
    """
    Return the vector representations of given quaternions, if the representations exist.
    :param q: q x 4 numpy array of quaternions
    :return: q x 3 numpy array of vector representations
    """
    assert q.dtype == np.float64
    assert np.all(q[..., 0] == 0)

    return q[..., 1:].copy()


def qvq_inverse(q: np.ndarray,
                v: np.ndarray) -> np.ndarray:
    """
    Compute the chained quaternion multiplication, q * v * q_inverse.
    :param q: 4-numpy-array representing a quaternion.
    :param v: v x 4 numpy array of quaternions.
    :return: v x 4 numpy array of resulting quaternions.
    """
    # assert v.dtype == np.float64 and axis.dtype == np.float64
    # assert v.shape == (3,) and axis.shape == (3,)
    #
    # rotation = rotation_quaternion(angle, axis)
    # return quaternion_multiplication(quaternion_multiplication(rotation,
    #                                                            vector_to_quaternion(v)),
    #                                  quaternion_inverse(rotation))[1:]
    assert v.dtype == np.float64 and q.dtype == np.float64
    assert q.shape == (4,) and v.shape[-1] == 4

    return v @ (np.array([[norm(q) ** 2, 0, 0, 0],
                          [0, norm(q[[0, 1]]) ** 2 - norm(q[[2, 3]]) ** 2, 2 * (q[1] * q[2] + q[0] * q[3]),
                           2 * (q[1] * q[3] - q[0] * q[2])],
                          [0, 2 * (q[1] * q[2] - q[0] * q[3]), norm(q[[0, 2]]) ** 2 - norm(q[[1, 3]]) ** 2,
                           2 * (q[2] * q[3] + q[0] * q[1])],
                          [0, 2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]),
                           norm(q[[0, 3]]) ** 2 - norm(q[[1, 2]]) ** 2]])
                / (np.linalg.norm(q) ** 2))
