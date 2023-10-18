import unittest
import numpy as np
import src.quaternion as s


class OrientationQuaternion(unittest.TestCase):
    def test_one_vector_one_reference(self) -> None:
        position = np.array([6., 3., 10.]) / np.linalg.norm(np.array([6., 3., 10.]))
        reference = np.array([2., 9., 8.]) / np.linalg.norm(np.array([2., 9., 8.]))
        expected = s.rotation_quaternion(0.627328929,
                                         np.array([-33 / np.sqrt(1861),
                                                   -14 / np.sqrt(1861),
                                                   24 / np.sqrt(1861)]))
        actual = s.orientation_quaternion(position, reference)
        self.assertTrue(np.allclose(actual, expected))

    def test_multiple_vectors_one_reference(self) -> None:
        positions = np.array([[6. / 12.041594578792296, 3. / 12.041594578792296, 10. / 12.041594578792296],
                              [4. / 9.486832980505138, 7. / 9.486832980505138, 5. / 9.486832980505138],
                              [-7. / 11.575836902790225, 6. / 11.575836902790225, 7. / 11.575836902790225]])

        reference = np.array([2. / 12.206555615733702, 9. / 12.206555615733702, 8. / 12.206555615733702])

        expected = np.array([[np.cos(0.627328929 / 2),
                              np.sin(0.627328929 / 2) * -66. / 86.27861844049197,
                              np.sin(0.627328929 / 2) * -28. / 86.27861844049197,
                              np.sin(0.627328929 / 2) * 48. / 86.27861844049197],
                             [np.cos(np.arccos(111 / (9.486832980505138 * 12.206555615733702)) / 2),
                              np.sin(np.arccos(111 / (9.486832980505138 * 12.206555615733702)) / 2) * 11. / 33.0,
                              np.sin(np.arccos(111 / (9.486832980505138 * 12.206555615733702)) / 2) * -22. / 33.0,
                              np.sin(np.arccos(111 / (9.486832980505138 * 12.206555615733702)) / 2) * 22. / 33.0],
                             [np.cos(np.arccos(96 / (11.575836902790225 * 12.206555615733702)) / 2),
                              np.sin(np.arccos(96 / (11.575836902790225 * 12.206555615733702)) / 2) * -15. / 103.6822067666386,
                              np.sin(np.arccos(96 / (11.575836902790225 * 12.206555615733702)) / 2) * 70. / 103.6822067666386,
                              np.sin(np.arccos(96 / (11.575836902790225 * 12.206555615733702)) / 2) * -75. / 103.6822067666386]])

        actual = s.orientation_quaternion(positions, reference)
        self.assertTrue(np.allclose(actual, expected))


class VectorToQuaternion(unittest.TestCase):
    def test_one_vector(self):
        vector = np.array([-3.58214914, 5.94684718, 6.93615019])
        quaternion = np.array([0.0, -3.58214914, 5.94684718, 6.93615019])

        self.assertTrue(np.array_equal(s.vector_to_quaternion(vector),
                                       quaternion))

    def test_multiple_vectors(self):
        vectors = np.array([[-9.63644587, -1.44091167, 2.84153074],
                            [6.75599579, 6.51768392, -8.04741763],
                            [7.09786417, 0.34134506, 8.39147551]])
        quaternions = np.array([[0.0, -9.63644587, -1.44091167, 2.84153074],
                                [0.0, 6.75599579, 6.51768392, -8.04741763],
                                [0.0, 7.09786417, 0.34134506, 8.39147551]])
        self.assertTrue(np.array_equal(s.vector_to_quaternion(vectors),
                                       quaternions))


class RotationQuaternion(unittest.TestCase):
    def setUp(self) -> None:
        self.axis = np.array([-3.58214914, 5.94684718, 6.93615019])
        self.axis_norm = 9.813611124439042
        self.angle = 0.6868663310182441
        self.axes = np.array([[-9.63644587, -1.44091167, 2.84153074],
                              [6.75599579, 6.51768392, -8.04741763],
                              [7.09786417, 0.34134506, 8.39147551]])
        self.axes_norm = np.array([10.149463650486522, 12.364651766514053, 10.996047174357063])
        self.angles = np.array([3.21870791, 7.74255517, 6.78019798])

    def test_one_axis_one_angle(self) -> None:
        expected = np.array([np.cos(self.angle / 2),
                             -np.sin(self.angle / 2) * self.axis[0] / self.axis_norm,
                             -np.sin(self.angle / 2) * self.axis[1] / self.axis_norm,
                             -np.sin(self.angle / 2) * self.axis[2] / self.axis_norm])

        self.assertTrue(np.allclose(s.rotation_quaternion(self.angle, self.axis),
                                    expected))

    def test_multiple_axes_one_angle(self) -> None:
        expected = np.array([[np.cos(self.angle / 2),
                              -np.sin(self.angle / 2) * self.axes[0, 0] / self.axes_norm[0],
                              -np.sin(self.angle / 2) * self.axes[0, 1] / self.axes_norm[0],
                              -np.sin(self.angle / 2) * self.axes[0, 2] / self.axes_norm[0]],
                             [np.cos(self.angle / 2),
                              -np.sin(self.angle / 2) * self.axes[1, 0] / self.axes_norm[1],
                              -np.sin(self.angle / 2) * self.axes[1, 1] / self.axes_norm[1],
                              -np.sin(self.angle / 2) * self.axes[1, 2] / self.axes_norm[1]],
                             [np.cos(self.angle / 2),
                              -np.sin(self.angle / 2) * self.axes[2, 0] / self.axes_norm[2],
                              -np.sin(self.angle / 2) * self.axes[2, 1] / self.axes_norm[2],
                              -np.sin(self.angle / 2) * self.axes[2, 2] / self.axes_norm[2]]])
        actual = s.rotation_quaternion(self.angle, self.axes)
        self.assertTrue(np.allclose(actual, expected))

    def test_multiple_axes_multiple_angles(self) -> None:
        expected = np.array([[np.cos(self.angles[0] / 2),
                              -np.sin(self.angles[0] / 2) * self.axes[0, 0] / self.axes_norm[0],
                              -np.sin(self.angles[0] / 2) * self.axes[0, 1] / self.axes_norm[0],
                              -np.sin(self.angles[0] / 2) * self.axes[0, 2] / self.axes_norm[0]],
                             [np.cos(self.angles[1] / 2),
                              -np.sin(self.angles[1] / 2) * self.axes[1, 0] / self.axes_norm[1],
                              -np.sin(self.angles[1] / 2) * self.axes[1, 1] / self.axes_norm[1],
                              -np.sin(self.angles[1] / 2) * self.axes[1, 2] / self.axes_norm[1]],
                             [np.cos(self.angles[2] / 2),
                              -np.sin(self.angles[2] / 2) * self.axes[2, 0] / self.axes_norm[2],
                              -np.sin(self.angles[2] / 2) * self.axes[2, 1] / self.axes_norm[2],
                              -np.sin(self.angles[2] / 2) * self.axes[2, 2] / self.axes_norm[2]]])
        actual = s.rotation_quaternion(self.angles, self.axes)
        self.assertTrue(np.allclose(actual, expected))

    def test_one_axis_multiple_angles(self) -> None:
        with self.assertRaises(AssertionError):
            expected = []
            self.assertTrue(np.allclose(s.rotation_quaternion(self.angles, self.axis),
                                        expected))


if __name__ == '__main__':
    unittest.main()
