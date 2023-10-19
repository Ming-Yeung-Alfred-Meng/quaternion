import numpy as np
from typing import Union
import src.quaternion as q


def quaternion_rotation(quaternion: np.ndarray,
                        vertices: np.ndarray) -> np.ndarray:
    """
    Rotate vertices using a quaternion.
    :param quaternion: 4-numpy-array defining a rotation.
    :param center: 3-numpy-array of the object's center.
    :param vertices: v x 3 numpy array of the object's vertices.
    :return: v x 3 numpy array of rotated vertices.
    """
    assert (quaternion.dtype == np.float64
            and vertices.dtype == np.float64)
    assert quaternion.shape == (4,)
    assert len(vertices.shape) <= 2 and vertices.shape[-1] == 3

    return q.quaternion_to_vector(q.qvq_inverse(quaternion,
                                                q.vector_to_quaternion(vertices)))


def rotation_about_x(center: np.ndarray, vertices: np.ndarray, degree: float) -> None:
    """
    Rotate (and mutate) vertices about the x-axis.
    :param center: 3D array of the center of the object vertices represent
    :param vertices: v x 3 matrix of vertices
    :param degree: degree in radian to rotate
    :return: None
    """
    vertices -= center
    vertices @= np.array([[1, 0, 0],
                          [0, np.cos(degree), -np.sin(degree)],
                          [0, np.sin(degree), np.cos(degree)]])
    vertices += center


def rotation_about_y(center: np.ndarray, vertices: np.ndarray, degree: float) -> None:
    """
    Rotate (and mutate) vertices about the y-axis.
    :param center: 3D array of the center of the object vertices represent
    :param vertices: v x 3 matrix of vertices
    :param degree: degree in radian to rotate
    :return: None
    """
    vertices -= center
    vertices @= np.array([[np.cos(degree), 0, np.sin(degree)],
                          [0, 1, 0],
                          [-np.sin(degree), 0, np.cos(degree)]])
    vertices += center


def rotation_about_z(center: np.ndarray, vertices: np.ndarray, degree: float) -> None:
    """
    Rotate (and mutate) vertices about the z-axis.
    :param center: 3D array of the center of the object vertices represent
    :param vertices: v x 3 matrix of vertices
    :param degree: degree in radian to rotate
    :return: None
    """
    vertices -= center
    vertices @= np.array([[np.cos(degree), -np.sin(degree), 0],
                          [np.sin(degree), np.cos(degree), 0],
                          [0, 0, 1]])
    vertices += center


def translation(center: np.ndarray,
                vertices: np.ndarray,
                step: Union[int, float],
                axis: int) -> None:
    """
    Translate (and mutate) vertices along an axis in the positive direction if step is positive, and vice versa.
    :param center: center of the object 'vertices' represents. This is mutated to the new translated center.
    :param vertices: n x 3 matrix of vertices in world cartesian coordinate
    :param step: number to translate by
    :param axis: axis along which to translate
    """
    assert len(center.shape) == 1
    assert len(center) == 3
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 3
    assert 0 <= axis <= 2

    vertices[:, axis] += step
    center[axis] += step


def uniform_scale(center: np.ndarray,
                  vertices: np.ndarray,
                  scale: Union[int, float]) -> None:
    """
    Uniformly scale (and mutate) vertices.
    :param center: center of the object 'vertices' represents
    :param vertices: v x 3 matrix of vertices
    :param scale: scale factor
    :return: None
    """
    assert 0 < scale

    vertices -= center
    vertices *= scale
    vertices += center
