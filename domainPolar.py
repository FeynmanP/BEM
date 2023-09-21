import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.misc import derivative


class SingleCavityPolar:
    """
    This class is for a domain including one singe cavity defined in polar system.
    When defining other different shapes by polar coordinates, inherit this class and modify the methods.
    Use D-shaped cavity for universality.
    """

    def __init__(self, size_params, N_elements, element_type='even', center=(0, 0)):
        """
        Initialize the SingleCavityPolar defined in polar system.
        :param size_params: Tuple. Size parameters to define the cavity.
        :param N_elements: Number of boundary elements
        :param center: Center of the cavity. Defaulted to (0, 0)
        """
        self._set_params(size_params)
        self.N_init = N_elements
        self.center = center
        self.element_type = element_type
        self.perimeter = self._compute_perimeter()
        self.element_info = self._generate_boundary_elements()
        self._adjust_center()

        # Length of element_info can be different from the given N.
        self.N = len(self.element_info)

    def __str__(self):
        return "D-shaped"
    def _set_params(self, size_params):
        """
        Set the parameters for the cavity.
        :param size_params:
        """
        self.R, self.d = size_params

    def boundary_r_theta(self, theta):
        # define the boundary from 0 to 2pi

        # Calculate the angle subtended by the chord at the center
        d = self.d
        R = self.R
        theta_0 = np.arccos(d / R)

        if 0 <= theta <= theta_0 or 2 * np.pi - theta_0 <= theta <= 2 * np.pi:
            return d / np.cos(theta if theta <= np.pi else 2 * np.pi - theta)

        # Outside this range, r is the radius of the circle
        return R

    def is_inside(self, x, y):
        x0, y0 = self.center[0], self.center[0]
        theta = np.arctan2(y - y0, x - x0)
        distance = np.sqrt((y - y0) ** 2 + (x - x0) ** 2)

        # theta is (-pi, pi), convert to (0, 2pi)
        theta_convert = (theta + 2 * pi) % 2 * pi
        if distance > self.boundary_r_theta(theta_convert):
            return False
        else:
            return True

    def _ds_dtheta(self, theta):
        return self._ds_dtheta_scipy(theta)

    def _ds_dtheta_scipy(self, theta):
        # use scipy.misc.derivative to calculate the derivative numerically.
        r_val = self.boundary_r_theta(theta)
        dr_dtheta_val = derivative(self.boundary_r_theta, theta, dx=1e-5)
        return np.sqrt(r_val ** 2 + dr_dtheta_val ** 2)

    def _dr_dtheta(self, theta):
        return derivative(self.boundary_r_theta, theta, dx=1e-5)

    def _d2r_dtheta2(self, theta):
        return derivative(self._dr_dtheta, theta, dx=1e-5)

    def _curvature_theta(self, theta):
        r_val = self.boundary_r_theta(theta)
        dr_dtheta_val = self._dr_dtheta(theta)
        d2r_dtheta2_val = self._d2r_dtheta2(theta)

        numerator = r_val ** 2 + 2 * dr_dtheta_val ** 2 - r_val * d2r_dtheta2_val
        denominator = (r_val ** 2 + dr_dtheta_val ** 2) ** 1.5

        return numerator / denominator

    def curvature_for_s(self, s):
        theta = self.find_theta_for_s(s)
        return self._curvature_theta(theta)

    def find_s_for_theta(self, theta):
        return self._s_theta_scipy(theta)

    def _s_theta_scipy(self, theta):
        # use scipy.integrate.quad to calculate the integral numerically.
        return quad(self._ds_dtheta, 0, theta)[0]

    def _compute_perimeter(self):
        return self.find_s_for_theta(2 * pi)

    def find_theta_for_s(self, s):
        # Actual arc length corresponding to fraction s
        s_actual = s * self.perimeter

        # Use root_scalar to find theta
        result = root_scalar(lambda theta: self.find_s_for_theta(theta) - s_actual, bracket=[0, 2 * pi])
        return result.root

    def _tangent_vector(self, theta):
        r_val = self.boundary_r_theta(theta)
        dr_dtheta_val = self._dr_dtheta(theta)
        Tx = dr_dtheta_val * np.cos(theta) - r_val * np.sin(theta)
        Ty = dr_dtheta_val * np.sin(theta) + r_val * np.cos(theta)
        magnitude = np.sqrt(Tx ** 2 + Ty ** 2)
        return Tx / magnitude, Ty / magnitude

    def _normal_vector(self, theta):
        Tx, Ty = self._tangent_vector(theta)
        return Ty, -Tx

    def _divide_boundary_evenly(self):
        """
        Divide the boundary evenly by the arc length s.
        s=1 for divide how
        :return: s values for startpoint, midpoint, endpoint and arclength for each element.
        """
        num_elements = self.N_init
        if not isinstance(num_elements, int) or num_elements <= 0:
            raise ValueError("N_init should be a positive integer.")

        # Use np.linspace method dividing boundary into num_elements+1 pieces
        s_values = np.linspace(0, 1, num_elements + 1)[:-1]
        s_values_midpoint = (s_values + 1 / (2 * num_elements)) % 1
        # s_values_midpoint = s_values
        s_values_startpoint = s_values
        s_values_endpoint = (s_values + 1 / num_elements) % 1

        # Calculate length for each element
        element_length = self.perimeter / num_elements

        return s_values_startpoint, s_values_midpoint, s_values_endpoint, element_length

    def _generate_boundary_elements(self):
        """
        Generate detailed information of boundary element used for BEM according to the shape and N_init.

        Notice: This method is only executable for shapes with vector-supported methods.
        Rewrite this method or do not inherit this class for boundary shapes without vector-supported methods.
        """
        if self.element_type == 'even':
            s_values_startpoint, s_values_midpoint, s_values_endpoint, element_length \
                = self._divide_boundary_evenly()
        else:
            # different space according to different curvatures, for example.
            print('Unsupported element type right now')
            return

        # Save detailed information for each element in one ndarray
        info_types = 13  # recording 13 different types of information for the boundary elements.
        element_info = np.zeros((len(s_values_midpoint), info_types))
        element_info[:, 0] = s_values_midpoint

        for i in range(len(element_info)):
            # compute and save (xi, yi)
            s_i = element_info[i, 0]
            s_i_start = s_values_startpoint[i]
            s_i_end = s_values_endpoint[i]

            theta_i = self.find_theta_for_s(s_i)
            r_i = self.boundary_r_theta(theta_i)

            element_info[i, 1] = r_i * np.cos(theta_i)
            element_info[i, 2] = r_i * np.sin(theta_i)

            # Save x and y components of outward normal vectors
            element_normal_outward_x_component = self._normal_vector(theta_i)[0]
            element_info[i, 3] = element_normal_outward_x_component
            element_normal_outward_y_component = self._normal_vector(theta_i)[1]
            element_info[i, 4] = element_normal_outward_y_component

            # Save x and y components of ccw tangent vectors
            element_tangent_ccw_x_component = self._tangent_vector(theta_i)[0]
            element_info[i, 5] = element_tangent_ccw_x_component
            element_tangent_ccw_y_component = self._tangent_vector(theta_i)[1]
            element_info[i, 6] = element_tangent_ccw_y_component

            element_info[i, 7] = self._curvature_theta(theta_i)

            theta_i_start = self.find_theta_for_s(s_i_start)
            r_i_start = self.boundary_r_theta(theta_i_start)
            theta_i_end = self.find_theta_for_s(s_i_end)
            r_i_end = self.boundary_r_theta(theta_i_end)
            element_info[i, 9] = r_i_start * np.cos(theta_i_start)
            element_info[i, 10] = r_i_start * np.sin(theta_i_start)
            element_info[i, 11] = r_i_end * np.cos(theta_i_end)
            element_info[i, 12] = r_i_end * np.sin(theta_i_end)

        element_info[:, 8] = element_length


        return element_info

    def _adjust_center(self):
        self.element_info[:, 1] += self.center[0]
        self.element_info[:, 2] += self.center[1]
        self.element_info[:, 9] += self.center[0]
        self.element_info[:, 10] += self.center[1]
        self.element_info[:, 11] += self.center[0]
        self.element_info[:, 12] += self.center[1]

def test():
    polarCavity = SingleCavityPolar((10, 5), 40)
    element_info = polarCavity.element_info
    # thetas = np.linspace(0, 2*pi, 1000)
    plt.plot(polarCavity.element_info[:, 1], polarCavity.element_info[:, 2])
    for i in range(len(element_info)):
        plt.plot([element_info[:, 1][i], element_info[:, 1][i] + element_info[:, 3][i]],
                 [element_info[:, 2][i], element_info[:, 2][i] + element_info[:, 4][i]])
        plt.plot([element_info[:, 1][i], element_info[:, 1][i] + element_info[:, 5][i]],
                 [element_info[:, 2][i], element_info[:, 2][i] + element_info[:, 6][i]])
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    test()