import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from numpy import pi


class SingleCavityCircle:
    """
    This class is for a domain including one singe cavity with circular boundary.
    When defining other different shapes, inherit this class and modify the methods.
    """
    def __init__(self, r, N, elements_type='even', center=(0, 0)):
        """
        Initialize the SingleCavityCircle with given parameters.
        :param r: Radius for the circular cavity.
        :param N: Number of boundary elements.
        :param elements_type: Way to divide the boundary.
        """
        self.cavity_size = r
        self.N = N
        self.center = center
        self.perimeter = self.compute_perimeter()
        self.element_type = elements_type
        self.element_info = self.generate_boundary_elements()

    def is_inside_cavity(self, x, y):
        """
        Check if a point (x, y) is inside the cavity.
        This method is for computing the wave function.
        :param x: x coordinate.
        :param y: y coordinate.
        :return: True for points inside the cavity.
        """
        r = self.cavity_size
        return x**2 + y**2 < r**2

    def compute_perimeter(self):
        return 2 * pi * self.cavity_size

    def tangent_normal_for_single_point(self, x, y):
        tangent, normal = np.array([-y, x]), np.array([x, y])
        return tangent / np.linalg.norm(tangent), normal / np.linalg.norm(normal)

    def tangent_normal(self, x, y):
        # Create tangent and normal vectors for each point
        tangent = np.stack([-y, x], axis=0)
        normal = np.stack([x, y], axis=0)

        # Compute the norm for each vector
        tangent_norm = np.linalg.norm(tangent, axis=0, keepdims=True)
        normal_norm = np.linalg.norm(normal, axis=0, keepdims=True)

        # Normalize each vector
        tangent_normalized = tangent / tangent_norm
        normal_normalized = normal / normal_norm

        return tangent_normalized, normal_normalized


    def curvature_s(self, s):
        """
        Compute curvature for given s
        (ONLY for circle)
        """
        r = self.cavity_size
        curvature = 1 / r
        if isinstance(s, np.ndarray):
            return np.vectorize(self.curvature_s)(s)
        return curvature

    def convert_s_xy(self, s):
        """convert s values (from 0 to 1) to the x, y coordinate"""
        # for circular boundary
        r = self.cavity_size
        theta = 2*pi * s
        x = np.cos(theta) * r
        y = np.sin(theta) * r

        return x, y

    def divide_boundary_evenly(self):
        """
        Divide the boundary evenly by the arc length s.
        s=1 for divide how
        :return: s values for startpoint, midpoint, endpoint and arclength for each element.
        """
        num_elements = self.N
        if not isinstance(num_elements, int) or num_elements <= 0:
            raise ValueError("N should be a positive integer.")

        # Use np.linspace method dividing boundary into num_elements+1 pieces
        s_values = np.linspace(0, 1, num_elements + 1)[:-1]
        s_values_midpoint = (s_values + 1 / (2 * num_elements)) % 1
        # s_values_midpoint = s_values
        s_values_startpoint = s_values
        s_values_endpoint = (s_values + 1 / num_elements) % 1

        # Calculate length for each element
        element_length = self.perimeter / num_elements

        return s_values_startpoint, s_values_midpoint, s_values_endpoint, element_length

    def generate_boundary_elements(self):
        """
        Generate detailed information of boundary element used for BEM according to the shape and N.

        Assumptions:
        - self.convert_s_xy and self.tangent_normal are vectorized functions.

        Notice: This method is only executable for shapes with vector-supported methods.
        Rewrite this method or do not inherit this class for boundary shapes without vector-supported methods.
        """
        if self.element_type == 'even':
            s_values_startpoint, s_values_midpoint, s_values_endpoint, element_length \
                = self.divide_boundary_evenly()
        else:
            # different space according to different curvatures, for example.
            print('Unsupported element type right now')

        # Generate coordinates and other properties for all elements

        # Position of startpoint, midpoint, and endpoint for each element.
        element_midpoint_x_values, element_midpoint_y_values = self.convert_s_xy(s_values_midpoint)
        element_startpoint_x_values, element_startpoint_y_values = self.convert_s_xy(s_values_startpoint)
        element_endpoint_x_values, element_endpoint_y_values = self.convert_s_xy(s_values_endpoint)

        element_tangent_ccw, element_normal_outward = self.tangent_normal(element_midpoint_x_values,
                                                                          element_midpoint_y_values)
        element_curvatures = self.curvature_s(s_values_midpoint)

        # Save detailed information for each element in one ndarray
        info_types = 13  # recording 13 different types of information for the boundary elements.
        element_info = np.zeros((len(s_values_midpoint), info_types))
        element_info[:, 0] = s_values_midpoint
        element_info[:, 1] = element_midpoint_x_values
        element_info[:, 2] = element_midpoint_y_values

        # Save x and y components of outward normal vectors
        element_normal_outward_x_component = element_normal_outward[0]
        element_info[:, 3] = element_normal_outward_x_component
        element_normal_outward_y_component = element_normal_outward[1]
        element_info[:, 4] = element_normal_outward_y_component

        # Save x and y components of ccw tangent vectors
        element_tangent_ccw_x_component = element_tangent_ccw[0]
        element_info[:, 5] = element_tangent_ccw_x_component
        element_tangent_ccw_y_component = element_tangent_ccw[1]
        element_info[:, 6] = element_tangent_ccw_y_component

        element_info[:, 7] = element_curvatures

        element_info[:, 8] = element_length

        element_info[:, 9] = element_startpoint_x_values
        element_info[:, 10] = element_startpoint_y_values
        element_info[:, 11] = element_endpoint_x_values
        element_info[:, 12] = element_endpoint_y_values

        return element_info


class SingleCavityStadium(SingleCavityCircle):
    def __init__(self, r, N):
        self.ratio = 1
        super().__init__(r, N)

    def is_inside_cavity(self, x, y):
        r = self.cavity_size
        d = r * self.ratio
        if np.abs(x) > d + r or np.abs(y) > r:
            return False
        elif (np.abs(x) - d)**2 + y**2 > r**2 and np.abs(x) > d:
            return False
        else:
            return True

    def compute_perimeter(self):
        r = self.cavity_size
        d = r * self.ratio
        perimeter = 2 * pi * r + 4 * d
        return perimeter

    def convert_s_xy(self, s):
        # only convert a quarter of the stadium-shapes, so 0<=s<=1/4
        r = self.cavity_size
        d = r * self.ratio
        length_quarter_circle = 0.5 * pi * r
        length_line = d
        frac_arc = length_quarter_circle / self.perimeter
        if s < frac_arc:
            theta = s * self.perimeter / r
            x = d + r * np.cos(theta)
            y = r * np.sin(theta)
            return x, y
        else:
            x = d - (s * self.perimeter - length_quarter_circle)
            y = r
            return x, y

    def tangent_normal_for_single_point(self, x, y):
        # outward norm and CCW tangent
        r = self.cavity_size
        d = r * self.ratio
        if x > d:
            # circle part
            theta = np.arctan2(y, x - d)
            norm_x = np.cos(theta)
            norm_y = np.sin(theta)
            tang_x = -np.sin(theta)
            tang_y = np.cos(theta)
        else:
            # line part
            norm_x = 0
            norm_y = 1
            tang_x = -1
            tang_y = 0
        return np.array([tang_x, tang_y]), np.array([norm_x, norm_y])


    def generate_boundary_elements(self):
        r = self.cavity_size
        d = r * self.ratio

        # consider 1/4 of the perimeter
        length_quarter_circle = 0.5 * pi * r
        length_line = d
        s_quarter_circle = length_quarter_circle / self.perimeter

        n = self.N // 4
        n_circle = int(n * length_quarter_circle / (0.25 * self.perimeter))
        n_line = n - n_circle

        s_circle_values = np.linspace(0, s_quarter_circle, n_circle + 1)[:-1]
        arc_length_circle = length_quarter_circle / n_circle
        s_circle_midpoints = s_circle_values + 0.5 * arc_length_circle / self.perimeter

        s_line_values = np.linspace(s_quarter_circle, 1/4, n_line + 1)[:-1]
        arc_length_line = length_quarter_circle / n_line
        s_line_midpoints = s_line_values + 0.5 * arc_length_line / self.perimeter

        # Save detailed information for each element in one ndarray
        info_types = 13  # recording 13 different types of information for the boundary elements.

        element_info_quarter_1 = np.zeros((n, info_types))
        element_info_quarter_1[:, 0] = np.concatenate((s_circle_midpoints, s_line_midpoints))

        for i in range(n):
            s_value = element_info_quarter_1[:, 0][i]
            x, y = self.convert_s_xy(s_value)
            element_info_quarter_1[:, 1][i] = x
            element_info_quarter_1[:, 2][i] = y

        for i in range(n):
            # Save x and y components of outward normal vectors
            x, y = element_info_quarter_1[:, 1][i], element_info_quarter_1[:, 2][i]
            element_tangent_ccw, element_normal_outward = self.tangent_normal_for_single_point(x, y)

            element_normal_outward_x_component = element_normal_outward[0]
            element_info_quarter_1[:, 3][i] = element_normal_outward_x_component
            element_normal_outward_y_component = element_normal_outward[1]
            element_info_quarter_1[:, 4][i] = element_normal_outward_y_component

            # Save x and y components of ccw tangent vectors
            element_tangent_ccw_x_component = element_tangent_ccw[0]
            element_info_quarter_1[:, 5][i] = element_tangent_ccw_x_component
            element_tangent_ccw_y_component = element_tangent_ccw[1]
            element_info_quarter_1[:, 6][i] = element_tangent_ccw_y_component

        for i in range(n_circle):
            element_info_quarter_1[:, 7][i] = 1 / r
            element_info_quarter_1[:, 8][i] = arc_length_circle

        for i in range(n_circle, n):
            element_info_quarter_1[:, 7][i] = 0
            element_info_quarter_1[:, 8][i] = arc_length_line

        # points in 2nd
        element_info_quarter_2 = element_info_quarter_1.copy()

        element_info_quarter_2 = element_info_quarter_2[::-1]
        element_info_quarter_2[:, 0] = 0.5 - element_info_quarter_1[:, 0]
        element_info_quarter_2[:, 1] = -element_info_quarter_2[:, 1]
        element_info_quarter_2[:, 2] = element_info_quarter_2[:, 2]
        element_info_quarter_2[:, 3] = - element_info_quarter_2[:, 3]
        element_info_quarter_2[:, 4] = element_info_quarter_2[:, 4]
        element_info_quarter_2[:, 5] = -element_info_quarter_2[:, 5]
        element_info_quarter_2[:, 6] = element_info_quarter_2[:, 6]

        # points in 3rd
        element_info_quarter_3 = element_info_quarter_1.copy()

        element_info_quarter_3[:, 0] = element_info_quarter_3[:, 0] + 0.5
        element_info_quarter_3[:, 1] = -element_info_quarter_3[:, 1]
        element_info_quarter_3[:, 2] = -element_info_quarter_3[:, 2]
        element_info_quarter_3[:, 3] = -element_info_quarter_3[:, 3]
        element_info_quarter_3[:, 4] = -element_info_quarter_3[:, 4]
        element_info_quarter_3[:, 5] = -element_info_quarter_3[:, 5]
        element_info_quarter_3[:, 6] = -element_info_quarter_3[:, 6]

        # points in 4th
        element_info_quarter_4 = element_info_quarter_2.copy()
        element_info_quarter_4[:, 0] = element_info_quarter_4[:, 0] + 0.5
        element_info_quarter_4[:, 1] = -element_info_quarter_4[:, 1]
        element_info_quarter_4[:, 2] = -element_info_quarter_4[:, 2]
        element_info_quarter_4[:, 3] = -element_info_quarter_4[:, 3]
        element_info_quarter_4[:, 4] = -element_info_quarter_4[:, 4]
        element_info_quarter_4[:, 5] = -element_info_quarter_4[:, 5]
        element_info_quarter_4[:, 6] = -element_info_quarter_4[:, 6]

        element_info = np.concatenate((element_info_quarter_1, element_info_quarter_2,
                                       element_info_quarter_3, element_info_quarter_4))

        return element_info


def test_generate_elements():
    cavity_test = SingleCavityCircle(10, 1000)
    element_info = cavity_test.element_info
    print(element_info.shape)
    thetas = np.linspace(0, 2*pi, 2048)
    plt.plot(np.cos(thetas), np.sin(thetas), '-', linewidth=0.5)
    plt.plot(element_info[:, 1], element_info[:, 2], '.')
    for i in range(len(element_info)):
        plt.plot([element_info[:, 1][i], element_info[:, 1][i] + element_info[:, 3][i]],
                 [element_info[:, 2][i], element_info[:, 2][i] + element_info[:, 4][i]])
    plt.axis('equal')
    plt.show()
    # DONE! 230827

def test_stadium():
    Stadium = SingleCavityStadium(1, 200)
    element_info = Stadium.element_info
    print(Stadium.element_info.shape)
    bdry_x_data = Stadium.element_info[:, 1]
    bdry_y_data = Stadium.element_info[:, 2]
    plt.plot(bdry_x_data, bdry_y_data, '.-')
    for i in range(len(element_info)):
        plt.plot([element_info[:, 1][i], element_info[:, 1][i] + element_info[:, 3][i]],
                 [element_info[:, 2][i], element_info[:, 2][i] + element_info[:, 4][i]])
        # plt.plot([element_info[:, 1][i], element_info[:, 1][i] + element_info[:, 5][i]],
        #          [element_info[:, 2][i], element_info[:, 2][i] + element_info[:, 6][i]])
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # test_generate_elements()
    test_stadium()

