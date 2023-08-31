import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel1
from numpy import pi


class SingleCavityCircle:
    def __init__(self, r, N, elements_type='even'):
        self.cavity_size = r
        self.N = N
        self.perimeter = self.compute_perimeter()
        self.element_type = elements_type
        self.element_info = self.generate_boundary_elements()


    def is_inside_cavity(self, x, y):
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

    def n_inside(self, x, y):
        return 3.3

    def n_outside(self, x, y):
        return 1

    def curvature_s(self, s):
        """
        Compute curvature for given s
        """
        r = self.cavity_size
        curvature = 1 / r
        if isinstance(s, np.ndarray):
            return np.vectorize(self.curvature_s)(s)
        return curvature

    def convert_s_xy(self, s):
        """convert s values (from 0 to 1) to the x, y coordinate"""
        # for circular boundary
        theta = 2*pi * s
        x = np.cos(theta)
        y = np.sin(theta)

        return x, y

    def divide_boundary_evenly(self):
        """
        Divide the boundary evenly by the arc length s.
        :return: s values for startpoint, midpoint, endpoint and arclength for each element.
        """
        num_elements = self.N
        if not isinstance(num_elements, int) or num_elements <= 0:
            raise ValueError("N should be a positive integer.")

        # Use np.linspace method dividing boundary into num_elements+1 pieces
        s_values = np.linspace(0, 1, num_elements + 1)[:-1]
        s_values_midpoint = (s_values + 1 / (2 * num_elements)) % 1
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
            # different space according to different curvatures for example
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
        element_info = np.zeros((len(s_values_midpoint), 13))
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


def test_generate_elements():
    cavity_test = SingleCavityCircle(1, 100)
    element_info = cavity_test.element_info
    print(element_info[1])
    thetas = np.linspace(0, 2*pi, 2048)
    plt.plot(np.cos(thetas), np.sin(thetas), '-', linewidth=0.5)
    plt.plot(element_info[:, 1], element_info[:, 2], '.')
    for i in range(len(element_info)):
        plt.plot([element_info[:, 1][i], element_info[:, 1][i] + element_info[:, 3][i]],
                 [element_info[:, 2][i], element_info[:, 2][i] + element_info[:, 4][i]])
    plt.axis('equal')
    plt.show()
    # DONE! 230827


if __name__ == '__main__':
    test_generate_elements()

