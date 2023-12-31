import sys
from typing import Tuple, Union, Dict, Callable

import pygame
import numpy as np

import src.coordinates as coc
import src.transformations as t
import src.quaternion as q
import src.interpolation as i


def animate_between_keyframes(screen: pygame.Surface,
                              camera_location: np.ndarray,
                              projection_method: int,
                              orientation_keyframes: np.ndarray,
                              center_keyframes: np.ndarray,
                              reference_vertices: np.ndarray,
                              faces: np.ndarray,
                              fps: int = 30,
                              number_of_frames: int = 100) -> bool:
    """
    Interpolate two keyframes and play animation.
    :param screen: PyGame display window.
    :param camera_location: 3-numpy-array of camera location in world frame.
    :param projection_method: 0 to use perspective projection, 1 to use orthographic projection.
    :param orientation_keyframes: 2 x 4 numpy array of object orientations of two keyframes in quaternions.
    If it has nan values, i.e. at least one keyframe has not been captured, the function exits and returns False.
    :param center_keyframes: 2 x 3 numpy array of object centers of two keyframes.
    If it has nan values, i.e. at least one keyframe has not been captured, the function exits and return False.
    :param reference_vertices: v x 3 object vertices with (0, 0, 0) as center. All orientations are relative to them.
    :param faces: 2D numpy array of face indices.
    :param fps: number of frames per second of the animation.
    :param number_of_frames: number of frames of the animation.
    :return: True if the animation was successfully created and played. False otherwise.
    """
    assert projection_method == 0 or projection_method == 1
    assert (orientation_keyframes.dtype == np.float64
            and center_keyframes.dtype == np.float64
            and camera_location.dtype == np.float64
            and reference_vertices.dtype == np.float64
            and faces.dtype == np.int64)
    assert (camera_location.shape == (3,)
            and orientation_keyframes.shape == (2, 4)
            and center_keyframes.shape == (2, 3)
            and len(reference_vertices.shape) <= 2 and reference_vertices.shape[-1] == 3
            and len(faces.shape) <= 2)

    if np.any(np.isnan(orientation_keyframes)) or np.any(np.isnan(center_keyframes)):
        return False

    interpolated_quaternions = i.linear_interpolation(orientation_keyframes,
                                                      number_of_frames)
    interpolated_centers = i.linear_interpolation(center_keyframes,
                                                  number_of_frames)

    clock = pygame.time.Clock()
    for j in range(number_of_frames):
        render(screen,
               t.quaternion_rotation(interpolated_quaternions[j], reference_vertices) + interpolated_centers[j],
               faces, projection(projection_method, camera_location))

        clock.tick(fps)

    return True


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
    Record orientation and center of a keyframe.
    :param orientation: 4-numpy-array of current orientation
    :param center: 3-numpy-array of the object's center in world frame
    :param orientation_keyframe: 4-numpy-array where orientation is stored.
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


def draw_wireframe(screen: pygame.Surface, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Draw the 2D projected image of the wireframe of a 3D mesh onto the PyGame surface.
    :param screen: pygame Surface
    :param vertices: n x 2 matrix of projected vertices of the wireframe
    :param faces: m x 4 matrix of face indices
    :return None
    """
    assert len(vertices.shape) == 2
    assert vertices.shape[1] == 2

    vertices_to_draw = coc.cartesian_to_pygame(vertices, screen.get_height())

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
           projection_function: Callable[[np.ndarray], np.ndarray]) -> None:
    """
    Render the environment in which the object 'vertices' and 'faces' represent.
    :param screen: PyGame display window
    :param vertices: v x 3 matrix of vertices
    :param faces: f x 4 matrix of face indices
    :param projection_function: a function that project 'vertices' onto 'screen'. Takes v x 3 matrix of vertices and
    outputs v x 2 matrix of projected vertices
    :return: None
    """
    screen.fill((255, 255, 255))

    draw_wireframe(screen, projection_function(vertices), faces)

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


def load_settings() -> Dict:
    """Load settings of the simulation."""
    center = np.array([300., 300., 300.])
    vertices, faces = init_cuboid(center, 50, 50, 50)
    return {"center": center,
            "vertices": vertices,
            "faces": faces,
            "reference_vertices": vertices - center,
            "orientation": np.array([1., 0., 0., 0.]),
            "orientation_keyframes": np.full((2, 4), np.nan, dtype=np.float64),
            "center_keyframes": np.full((2, 3), np.nan, dtype=np.float64),
            "screen_width": 800,
            "screen_height": 800,
            "camera_location": np.array([400., 400., -100.]),
            "rotation_speed": np.pi / 80.,
            "translation_speed": 5.,
            "fps": 60,
            "number_of_frames": 100,
            "projection_method": 1}


def load_translation_controls(center: np.ndarray,
                              vertices: np.ndarray,
                              translation_speed: float) -> Dict:
    """
    Return a dictionary of bindings between keys and scripts for translating an object.
    :param center: center of the object 'vertices' represents. This is mutated to the new translated center.
    :param vertices: n x 3 matrix of vertices in world cartesian coordinate
    :param translation_speed: translation speed.
    """
    return {pygame.K_LEFT: (t.translation, (center, vertices, - translation_speed, 0)),
            pygame.K_RIGHT: (t.translation, (center, vertices, translation_speed, 0)),
            pygame.K_UP: (t.translation, (center, vertices, translation_speed, 1)),
            pygame.K_DOWN: (t.translation, (center, vertices, - translation_speed, 1)),
            pygame.K_w: (t.translation, (center, vertices, translation_speed, 2)),
            pygame.K_s: (t.translation, (center, vertices, - translation_speed, 2))}


def load_rotation_controls(center: np.ndarray,
                           vertices: np.ndarray,
                           orientation: np.ndarray,
                           rotation_speed: float) -> Dict:
    """
    Return a dictionary of bindings between keys and scripts for rotating the object.
    :param center: 3-numpy-array of center of the object. Mutated after rotation.
    :param vertices: v x 3 numpy array of the object vertices. Mutated after rotation.
    :param orientation: 4-numpy-array of the object's orientation in quaternion.
    :param rotation_speed: rotation speed.
    :return: None
    """
    return {pygame.K_LEFT: (rotation_script, (center, vertices, - rotation_speed, orientation, 1)),
            pygame.K_RIGHT: (rotation_script, (center, vertices, rotation_speed, orientation, 1)),
            pygame.K_UP: (rotation_script, (center, vertices, - rotation_speed, orientation, 0)),
            pygame.K_DOWN: (rotation_script, (center, vertices, rotation_speed, orientation, 0)),
            pygame.K_a: (rotation_script, (center, vertices, - rotation_speed, orientation, 2)),
            pygame.K_d: (rotation_script, (center, vertices, rotation_speed, orientation, 2))}


def load_record_keyframes_controls(center: np.ndarray,
                                   orientation: np.ndarray,
                                   orientation_keyframes: np.ndarray,
                                   center_keyframes: np.ndarray) -> Dict:
    """
    Return a dictionary of bindings between keys and scripts for recording orientation and center of a keyframe.
    :param orientation: 4-numpy-array of current orientation
    :param center: 3-numpy-array of the object's center in world frame
    :param orientation_keyframes: 4-numpy-array where orientation is stored.
    :param center_keyframes: 3-numpy-array where current center is stored.
    :return: None
    """
    return {pygame.K_i: (record_keyframes, (orientation,
                                            center,
                                            orientation_keyframes[0],
                                            center_keyframes[0])),
            pygame.K_o: (record_keyframes, (orientation,
                                            center,
                                            orientation_keyframes[1],
                                            center_keyframes[1]))}


def load_play_keyframe_animation_controls(screen,
                                          camera_location: np.ndarray,
                                          projection_method: int,
                                          orientation_keyframes,
                                          center_keyframes,
                                          reference_vertices,
                                          faces,
                                          fps,
                                          number_of_frames) -> Dict:
    """
    Return a dictionary of bindings between keys and scripts for animating the interpolated frames.
    :param screen: PyGame display window.
    :param camera_location: 3-numpy-array of camera location in world frame.
    :param projection_method: 0 to use perspective projection, 1 to use orthographic projection.
    :param orientation_keyframes: 2 x 4 numpy array of object orientations of two keyframes in quaternions.
    If it has nan values, i.e. at least one keyframe has not been captured, the function exits and returns False.
    :param center_keyframes: 2 x 3 numpy array of object centers of two keyframes.
    If it has nan values, i.e. at least one keyframe has not been captured, the function exits and return False.
    :param reference_vertices: v x 3 object vertices with (0, 0, 0) as center. All orientations are relative to them.
    :param faces: 2D numpy array of face indices.
    :param fps: number of frames per second of the animation.
    :param number_of_frames: number of frames of the animation.
    :return: None
    """
    return {pygame.K_p: (play_keyframe_animation_script, (screen,
                                                          camera_location,
                                                          projection_method,
                                                          orientation_keyframes,
                                                          center_keyframes,
                                                          reference_vertices,
                                                          faces,
                                                          fps,
                                                          number_of_frames))}


def play_keyframe_animation_script(screen,
                                   camera_location: np.ndarray,
                                   projection_method: int,
                                   orientation_keyframes,
                                   center_keyframes,
                                   reference_vertices,
                                   faces,
                                   fps,
                                   number_of_frames) -> None:
    """
    Script to execute when animating the interpolated frames.
    :param screen: PyGame display window.
    :param camera_location: 3-numpy-array of camera location in world frame.
    :param projection_method: 0 to use perspective projection, 1 to use orthographic projection.
    :param orientation_keyframes: 2 x 4 numpy array of object orientations of two keyframes in quaternions.
    If it has nan values, i.e. at least one keyframe has not been captured, the function exits and returns False.
    :param center_keyframes: 2 x 3 numpy array of object centers of two keyframes.
    If it has nan values, i.e. at least one keyframe has not been captured, the function exits and return False.
    :param reference_vertices: v x 3 object vertices with (0, 0, 0) as center. All orientations are relative to them.
    :param faces: 2D numpy array of face indices.
    :param fps: number of frames per second of the animation.
    :param number_of_frames: number of frames of the animation.
    :return: None
    """
    if animate_between_keyframes(screen, camera_location, projection_method, orientation_keyframes,
                                 center_keyframes,
                                 reference_vertices, faces, fps, number_of_frames):
        orientation_keyframes[:] = np.full_like(orientation_keyframes, np.nan, dtype=np.float64)
        center_keyframes[:] = np.full_like(orientation_keyframes, np.nan, dtype=np.float64)


def rotation_script(center: np.ndarray,
                    vertices: np.ndarray,
                    rotation_speed: float,
                    orientation: np.ndarray,
                    axis: int) -> None:
    """
    Script to execute when rotating the object.
    :param center: 3-numpy-array of center of the object. Mutated after rotation.
    :param vertices: v x 3 numpy array of the object vertices. Mutated after rotation.
    :param rotation_speed: rotation speed.
    :param orientation: 4-numpy-array of the object's orientation in quaternion.
    :param axis: canonical axis about which to rotate. 0, 1, 2 for x-, y-, and z-axis, respectively.
    :return: None
    """
    assert axis == 0 or axis == 1 or axis == 2

    if axis == 0:
        t.rotation_about_x(center, vertices, rotation_speed)
        orientation[:] = update_orientation(orientation, rotation_speed, np.array([1., 0., 0.]))
    elif axis == 1:
        t.rotation_about_y(center, vertices, rotation_speed)
        orientation[:] = update_orientation(orientation, rotation_speed, np.array([0., 1., 0.]))
    else:
        t.rotation_about_z(center, vertices, rotation_speed)
        orientation[:] = update_orientation(orientation, rotation_speed, np.array([0., 0., 1.]))


def start_environment(screen: pygame.Surface,
                      settings: Dict,
                      rotation_controls: Dict,
                      translation_controls: Dict,
                      record_keyframes_controls: Dict,
                      play_keyframe_animation_controls: Dict) -> None:
    """
    The main loop of the simulation.
    :param screen: pygame screen/display.
    :param settings: dictionary of settings
    :param rotation_controls: dictionary of bindings between keys and scripts for the rotation of the object.
    :param translation_controls: dictionary of bindings between keys and scripts for the translation of the object.
    :param record_keyframes_controls: dictionary of bindings between keys and scripts for recording keyframes.
    :param play_keyframe_animation_controls: dictionary of bindings between keys and scripts for animating the
    interpolated frames.
    :return: None
    """
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        control(record_keyframes_controls)
        control(play_keyframe_animation_controls)

        if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
            control(rotation_controls)

        else:
            control(translation_controls)

        render(screen,
               settings["vertices"],
               settings["faces"],
               projection(settings["projection_method"],
                          settings["camera_location"]))

    pygame.quit()
    sys.exit(0)


def run() -> None:
    """
    Run the simulation.
    :return: None
    """
    settings = load_settings()

    pygame.init()

    screen = pygame.display.set_mode((settings["screen_width"], settings["screen_height"]))
    pygame.display.set_caption('Quaternion')

    translation_controls = load_translation_controls(settings["center"],
                                                     settings["vertices"],
                                                     settings["translation_speed"])

    rotation_controls = load_rotation_controls(settings["center"],
                                               settings["vertices"],
                                               settings["orientation"],
                                               settings["rotation_speed"])

    record_keyframes_controls = load_record_keyframes_controls(settings["center"],
                                                               settings["orientation"],
                                                               settings["orientation_keyframes"],
                                                               settings["center_keyframes"])

    play_keyframe_animation_controls = load_play_keyframe_animation_controls(screen,
                                                                             settings["camera_location"],
                                                                             settings["projection_method"],
                                                                             settings["orientation_keyframes"],
                                                                             settings["center_keyframes"],
                                                                             settings["reference_vertices"],
                                                                             settings["faces"],
                                                                             settings["fps"],
                                                                             settings["number_of_frames"])
    start_environment(screen,
                      settings,
                      rotation_controls,
                      translation_controls,
                      record_keyframes_controls,
                      play_keyframe_animation_controls)