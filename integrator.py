import numpy as np
from scipy.special import hankel1
import matplotlib.pyplot as plt
from scipy.integrate import quad
import domain

from numpy import euler_gamma
from numpy import pi


def hankel1_visualize():
    x = np.linspace(-10, 10, 100)

    # plot 0th-order hankel1
    plt.plot(x, hankel1(0, x), 'r-')

    plt.axis('equal')
    plt.show()


def B_si_sj(r_i, r_j, r_start_j, r_end_j, kappa, n, k):
    """
    Compute integral kernels for the matrix elements in B_ij
    :param r_i: Position for fixed point.
    :param r_j: Position for the point which affect the fixed point r_i.
    :param r_start_j: Start point of arc s_j(r_j)
    :param r_end_j: End point of arc s_j(r_j)
    :param kappa: Curvature of arc at midpoint
    :param n: Refractive index
    :param k: Complex wave number
    :return:
    """
    x_i, y_i = r_i[0], r_i[1]
    x_j, y_j = r_j[0], r_j[1]

    x_start, y_start = r_start_j[0], r_start_j[1]  # start point of arc
    x_end, y_end = r_end_j[0], r_end_j[1]  # end point of arc
    radius = 1 / kappa

    def gaussian_rj(x, y):
        gaussian = -1j / 4 * hankel1(0, n * k * np.sqrt((x - x_i) ** 2 + (y - y_i) ** 2))
        return gaussian


def B_si_sj_circle(r_i, r_j, r_start_j, r_end_j, kappa, arc_length, n, k):
    """
    Compute integral kernels for the matrix elements in B_ij
    For SPECIAL case CIRCLE centered in (0, 0) ONLY!!
    :param r_i: Position for fixed point.
    :param r_j: Position for the point which affect the fixed point r_i.
    :param r_start_j: Start point of arc s_j(r_j)
    :param r_end_j: End point of arc s_j(r_j)
    :param kappa: Curvature of arc at midpoint
    :param arc_length: length for boundary element
    :param n: Refractive index
    :param k: Complex wave number
    :return:
    """
    x_i, y_i = r_i[0], r_i[1]
    x_j, y_j = r_j[0], r_j[1]

    x_start, y_start = r_start_j[0], r_start_j[1]  # start point of arc
    x_end, y_end = r_end_j[0], r_end_j[1]  # end point of arc
    radius = 1 / kappa

    def x(t):
        return radius * np.cos(t)

    def y(t):
        return radius * np.sin(t)

    def B_ri(x, y):
        gaussian = -1j / 4 * hankel1(0, n * k * np.sqrt((x - x_i) ** 2 + (y - y_i) ** 2))

        return -2 * gaussian

    def integrand_real(t):
        return np.real(B_ri(x(t), y(t))) * radius

    def integrand_imag(t):
        return np.imag(B_ri(x(t), y(t))) * radius

    theta1 = np.arctan2(y_start, x_start)
    theta2 = theta1 + arc_length / radius

    result_real, err_real = quad(integrand_real, theta1, theta2)
    result_imag, err_imag = quad(integrand_imag, theta1, theta2)

    return result_real + 1j * result_imag


def C_si_sj_circle(r_i, r_j, r_start_j, r_end_j, kappa, arc_length, n, k):
    """
    Compute integral kernels for the matrix elements in B_ij
    For SPECIAL case CIRCLE centered in (0, 0) ONLY!!
    :param r_i: Position for fixed point.
    :param r_j: Position for the point which affect the fixed point r_i.
    :param r_start_j: Start point of arc s_j(r_j)
    :param r_end_j: End point of arc s_j(r_j)
    :param kappa: Curvature of arc at midpoint
    :param arc_length: length for boundary element
    :param n: Refractive index
    :param k: Complex wave number
    :return:
    """
    x_i, y_i = r_i[0], r_i[1]
    x_j, y_j = r_j[0], r_j[1]

    x_start, y_start = r_start_j[0], r_start_j[1]  # start point of arc
    x_end, y_end = r_end_j[0], r_end_j[1]  # end point of arc
    radius = 1 / kappa

    def x(t):
        return radius * np.cos(t)

    def y(t):
        return radius * np.sin(t)

    def C_ri(x, y):
        r = np.array([x, y])
        normal_outward = np.array([x, y])
        normal_outward = normal_outward / np.linalg.norm(normal_outward)
        cos_alpha = normal_outward.dot(r - r_i) / np.linalg.norm(r - r_i)
        d_gaussian = (1j * n * k * cos_alpha) / 4 * hankel1(1, n * k * np.sqrt((x - x_i) ** 2 + (y - y_i) ** 2))
        return 2 * d_gaussian

    def integrand_real(t):
        return np.real(C_ri(x(t), y(t))) * radius

    def integrand_imag(t):
        return np.imag(C_ri(x(t), y(t))) * radius

    theta1 = np.arctan2(y_start, x_start)
    theta2 = theta1 + arc_length / radius

    result_real, err_real = quad(integrand_real, theta1, theta2)
    result_imag, err_imag = quad(integrand_imag, theta1, theta2)

    return result_real + 1j * result_imag


def B_ll(n, s_length, k):
    """
    Compute diagonal element of B for r'->r
    :param n: refractive index for domain
    :param s_length: element length
    :return: Diagonal element of B
    """
    B_ll = s_length / pi * (1 - np.log((n * k * s_length) / 4) + 1j * pi / 2 - euler_gamma)
    return B_ll


def C_ll(s_length, kappa):
    """
    Compute diagonal element of B for r'->r.
    :param s_length: element length.
    :param kappa: curvature for element.
    :return: Diagonal element of B
    """
    C_ll = -1 + kappa * s_length / (2 * pi)
    return C_ll


def compute_matrix(num_element, radius, k):
    # For TM-mode

    Domain = domain.SingleCavityCircle(radius, num_element)

    B1 = np.zeros((num_element, num_element), dtype=complex)
    C1 = np.zeros((num_element, num_element), dtype=complex)
    B2 = np.zeros((num_element, num_element), dtype=complex)
    C2 = np.zeros((num_element, num_element), dtype=complex)
    element_info = Domain.element_info
    n1 = 3.3
    n2 = 1

    # compute B1 and C1 (along the outside boundary in CCW direction)
    for i in range(len(element_info)):
        for j in range(len(element_info)):
            if i == j:
                s_length = element_info[j][8]
                kappa = element_info[j][7]
                B1[i][j] = B_ll(n1, s_length, k)
                C1[i][j] = C_ll(s_length, kappa)
            else:
                r_i = np.array([element_info[i][1], element_info[i][2]])
                r_j = np.array([element_info[j][1], element_info[j][2]])
                r_start_j = np.array([element_info[j][9], element_info[j][10]])
                r_end_j = np.array([element_info[j][11], element_info[j][12]])
                s_length_j = element_info[j][8]
                kappa = element_info[j][7]

                B1[i][j] = B_si_sj_circle(r_i, r_j, r_start_j, r_end_j, kappa, s_length_j, n1, k)
                C1[i][j] = C_si_sj_circle(r_i, r_j, r_start_j, r_end_j, kappa, s_length_j, n1, k)

    # change to CW direction, which is still ccw direction if consider the interior of domain1 is exterior

    element_info = element_info[::-1]

    for i in range(len(element_info)):
        for j in range(len(element_info)):
            if i == j:
                s_length = element_info[j][8]
                kappa = -element_info[j][7]  # negative curvature
                B2[i][j] = B_ll(n1, s_length, k)
                C2[i][j] = C_ll(s_length, kappa)
            else:
                r_i = np.array([element_info[i][1], element_info[i][2]])
                r_j = np.array([element_info[j][1], element_info[j][2]])
                r_start_j = np.array([element_info[j][9], element_info[j][10]])
                r_end_j = np.array([element_info[j][11], element_info[j][12]])
                s_length_j = element_info[j][8]
                kappa = -element_info[j][7]  # negative curvature

                B2[i][j] = B_si_sj_circle(r_i, r_j, r_start_j, r_end_j, kappa, s_length_j, n2, k)
                C2[i][j] = C_si_sj_circle(r_i, r_j, r_start_j, r_end_j, kappa, s_length_j, n2, k)

    matrix_k = np.block([[B1, C1], [B2, C2]])
    determinant = np.linalg.det(matrix_k)




    return determinant


def compute_results():
    """
    Compute the determinant for different k and save the absolute value of them in a 2d array.
    :return:
    """
    flag = 1
    k_real_values = np.linspace(49, 49.3, 30 + 1)
    k_imag_values = np.linspace(-0.02, -0.01, 10 + 1)
    num_element = 20
    radius = 1
    mat_result = np.zeros((len(k_real_values), len(k_imag_values), 3))
    for i in range(len(k_real_values)):
        for j in range(len(k_imag_values)):
            k = k_real_values[i] + 1j * k_imag_values[j]
            determinant = compute_matrix(num_element, radius, k)
            mat_result[i][j][0] = k_real_values[i]
            mat_result[i][j][1] = k_imag_values[j]
            mat_result[i][j][2] = np.abs(determinant)
            print(flag)
            flag += 1

    np.save('data_try_det_0830.npy', mat_result)

def visualize_det():
    data = np.load('data_try_det_0831_2.npy')
    X = data[:, :, 0]
    Y = data[:, :, 1]
    Z = data[:, :, 2]

    plt.pcolor(X, Y, Z)

    plt.show()




if __name__ == '__main__':
    compute_results()
    # compute_matrix(10-1j*0.001)
    # visualize_det()