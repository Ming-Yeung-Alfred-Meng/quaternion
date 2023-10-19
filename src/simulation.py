import sys
from typing import Tuple, Union, Dict, Callable

import pygame
import numpy as np

import src.coordinates as coc
import src.transformations as t
import src.quaternion as q


def update_orientation(orientation: np.ndarray,
                       angle: float,
                       axis: np.ndarray) -> np.ndarray:
    """
    Return the new orientation in quaternion after rotating about an axis by an angle.
    :param orientation: 4-numpy-array of orientation in quaternion before rotation.
    :param angle: radian angle of rotation.
    :param axis: 3-numpy-array of axis of rotation.
    :return: 4-numpy-array of orientation in quaternion after rotation
    """
    assert orientation.dtype == np.float64 and axis.dtype == np.float64
    assert orientation.shape == (4,) and axis.shape == (3,)

    return q.quaternion_multiplication(q.rotation_quaternion(angle, axis),
                                       orientation)


def record_keyframes(orientation: np.ndarray,
                     center: np.ndarray,
                     orientation_keyframe: np.ndarray,
                     center_keyframe: np.ndarray) -> None:
    """
    Get a vertex's position local to a given origin.
    :param orientation:
    :param vertex: 3-numpy-array in world frame used to determine orientation.
    :param center: 3-numpy-array of the object's center in world frame
    :param orientation_keyframe: 3-numpy-array where orientation is stored.
    :param center_keyframe: 3-numpy-array where current center is stored.
    :return: None
    """
    assert (orientation.dtype == np.float64
            and center.dtype == np.float64
            and orientation_keyframe.dtype == np.float64
            and center_keyframe.dtype == np.float64)
    assert orientation.shape == (4,) and center.shape == (3,)
    assert (orientation.shape == orientation_keyframe.shape
            and center.shape == center_keyframe.shape)

    orientation_keyframe[:] = orientation
    center_keyframe[:] = center


def init_cuboid(center: np.ndarray,
                half_width: Union[int, float],
                half_height: Union[int, float],
                half_depth: Union[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize a 3D cuboid.
    :param center: 3D float64 array of the center of the cuboid
    :param half_width: half of the width of the cuboid, i.e. length in x
    :param half_height: half of the height of the cuboid, i.e. length in y
    :param half_depth: half of the depth of the cuboid, i.e. length in z
    :return: 8 x 3 matrix of vertices of the cuboid, 6 x 4 matrix of face indices
    """
    assert len(center.shape) == 1
    assert len(center) == 3
    assert center.dtype == np.float64

    x, y, z = np.meshgrid((center[0] + half_width, center[0] - half_width),
                          (center[1] + half_height, center[1] - half_height),
                          (center[2] + half_depth, center[2] - half_depth), copy=True)

    assert x.dtype == np.float64

    return (np.column_stack((x.flatten(), y.flatten(), z.flatten())),
            np.array([[0, 4, 6, 2],
                      [1, 3, 7, 5],
                      [0, 2, 3, 1],
                      [0, 1, 5, 4],
                      [1, 3, 7, 5],
                      [2, 6, 7, 3]]))


def perspective_projection(camera_location: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Perspective projection of 3D vertices in world cartesian coordinate to the frame of the x-y plane.
    :param camera_location: array of camara location in world frame
    :param vertices: n x 3 matrix of vertices in world frame
    :return: n x 2 matrix of vertices in the frame of the x-y plane
    """
    assert len(camera_location.shape) == 1
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 3

    return coc.homogeneous_to_cartesian(coc.cartesian_to_homogeneous(vertices) @ np.array([[-camera_location[2], 0, 0],
                                                                                           [0, -camera_location[2], 0],
                                                                                           [camera_location[0],
                                                                                            camera_location[1], 1],
                                                                                           [0, 0,
                                                                                            -camera_location[2]]]))


def orthographic_projection(vertices: np.ndarray) -> np.ndarray:
    """
    Orthographic projection of 3D vertices in world cartesian coordinate to the frame of the x-y plane.
    :param vertices: n x 3 vertices to project
    :return: n x 2 projected vertices, i.e. **VIEW** into first two columns of 'vertices'
    """
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 3

    return vertices[:, 0:2]


def draw_wireframe(screen: pygame.Surface, vertices: np.ndarray, faces: np.ndarray, screen_height: int) -> None:
    """
    Draw the 2D projected image of the wireframe of a 3D mesh onto the PyGame surface.
    :param screen: pygame Surface
    :param vertices: n x 2 matrix of projected vertices of the wireframe
    :param faces: m x 4 matrix of face indices
    :param screen_height: height of the surface
    :return None
    """
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 2

    vertices_to_draw = coc.cartesian_to_pygame(vertices, screen_height)

    for face in faces:
        for i in range(face.shape[0]):
            pygame.draw.aaline(screen, (0, 0, 0),
                               vertices_to_draw[face[i]],
                               vertices_to_draw[face[(i + 1) % face.shape[0]]])


def control(key_to_action: Dict[int, Tuple[Callable, Tuple]]) -> None:
    """
    Execute the associated function when a key is pressed.
    :param key_to_action: A dictionary where keys are PyGame keys and values are tuples where the first element is the
    function to execute, and the second is a tuple of arguments into the function.
    :return: None
    """
    for key, action in key_to_action.items():
        if pygame.key.get_pressed()[key]:
            action[0](*action[1])


def render(screen: pygame.Surface,
           vertices: np.ndarray,
           faces: np.ndarray,
           screen_height: Union[int, float],
           projection_function: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    Render the environment in which the object 'vertices' and 'faces' represent.
    :param screen: PyGame display window
    :param vertices: v x 3 matrix of vertices
    :param faces: f x 4 matrix of face indices
    :param screen_height: height of the PyGame display window
    :param projection_function: a function that project 'vertices' onto 'screen'. Takes v x 3 matrix of vertices and
    outputs v x 2 matrix of projected vertices
    :return: None
    """
    screen.fill((255, 255, 255))

    draw_wireframe(screen, projection_function(vertices), faces, screen_height)

    pygame.display.update()


def projection(choice: int,
               camera_location: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a projection function based on 'choice'.
    :param choice: 0 for perspective projection, 1 for orthographic projection
    :param camera_location: 3D vector of camera center
    :return: a projection function that takes v x 3 matrix of vertices as input and output v x 2 matrix of projected
    vertices
    """
    assert choice == 0 or choice == 1
    assert len(camera_location.shape) == 1
    assert len(camera_location) == 3
    assert camera_location.dtype == np.float64

    if choice == 0:
        def perspective_projection_fixed_camera(vertices: np.ndarray) -> np.ndarray:
            return perspective_projection(camera_location, vertices)

        return perspective_projection_fixed_camera

    elif choice == 1:
        return orthographic_projection


def start_environment(screen_width: Union[int, float],
                      screen_height: Union[int, float],
                      camera_location: np.ndarray,
                      translation_controls: Dict[int, Tuple[Callable, Tuple]],
                      rotation_controls: Dict[int, Tuple[Callable, Tuple]],
                      uniform_scale_controls: Dict[int, Tuple[Callable, Tuple]],
                      vertices: np.ndarray,
                      faces: np.ndarray) -> None:
    """
    Begin the game loop.
    :param screen_width: width of the PyGame window
    :param screen_height: height of the PyGame window
    :param translation_controls: A dictionary specifying controls for translation of the object. Keys are PyGame keys
    and values are tuples where the first element is the function to execute, and the second is a tuple of arguments
    into the function.
    :param rotation_controls: A dictionary specifying controls for rotation of the object. Keys are PyGame keys and
    values are tuples where the first element is the function to execute, and the second is a tuple of arguments into
    the function.
    :param uniform_scale_controls: A dictionary specifying controls for uniform scale of the object. Keys are PyGame
    keys and values are tuples where the first element is the function to execute, and the second is a tuple of
    arguments into the function.
    :param camera_location:
    :param vertices: v x 3 matrix of vertices of the object to display
    :param faces: f x 4 matrix of vertices of face indices of the object
    :return: None
    """
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 3
    assert len(faces.shape) == 2
    assert faces.shape[1] == 4

    pygame.init()

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Affine Transformations')

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
            control(rotation_controls)

        else:
            control(translation_controls)
            control(uniform_scale_controls)

        render(screen, vertices, faces, screen_height, projection(0, camera_location))

    pygame.quit()
    sys.exit(0)


def run() -> None:
    center = np.array([300., 300., 300.])
    vertices, faces = init_cuboid(center, 50, 50, 50)

    translation_controls = {pygame.K_LEFT: (t.translation, (center, vertices, - 5, 0)),
                            pygame.K_RIGHT: (t.translation, (center, vertices, 5, 0)),
                            pygame.K_UP: (t.translation, (center, vertices, 5, 1)),
                            pygame.K_DOWN: (t.translation, (center, vertices, - 5, 1)),
                            pygame.K_w: (t.translation, (center, vertices, 5, 2)),
                            pygame.K_s: (t.translation, (center, vertices, - 5, 2))}

    rotation_controls = {pygame.K_LEFT: (t.rotation_about_y, (center, vertices, - np.pi / 80)),
                         pygame.K_RIGHT: (t.rotation_about_y, (center, vertices, np.pi / 80)),
                         pygame.K_UP: (t.rotation_about_x, (center, vertices, - np.pi / 80)),
                         pygame.K_DOWN: (t.rotation_about_x, (center, vertices, np.pi / 80)),
                         pygame.K_a: (t.rotation_about_z, (center, vertices, - np.pi / 80)),
                         pygame.K_d: (t.rotation_about_z, (center, vertices, np.pi / 80))}

    uniform_scale_controls = {pygame.K_n: (t.uniform_scale, (center, vertices, 0.9)),
                              pygame.K_m: (t.uniform_scale, (center, vertices, 1.1))}

    start_environment(800, 800, np.array([400., 400., -100.]),
                      translation_controls,
                      rotation_controls,
                      uniform_scale_controls,
                      vertices, faces)
