import numpy as np
import domain
from scipy.special import hankel1
from numpy import pi, euler_gamma
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter
import time
from scipy.linalg import lu_factor, det

"""
This program calculates the 2N*2N matrix [Bi, Ci] for BEM.
"""

def B_si_sj_approx(x_i, y_i, x_j, y_j, norm_x_j, norm_y_j, arc_length_j, n, k):
    """
    Compute integral kernels for the matrix elements in B_ij
    Approximate the integral by assuming the invariance of integral kernel along the arc.
    """
    distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
    z = n * k * distance
    gaussian = (-1j / 4) * hankel1(0, z)

    result = -2 * gaussian * arc_length_j

    return result


def C_si_sj_approx(x_i, y_i, x_j, y_j, norm_x_j, norm_y_j, arc_length_j, n, k):
    """
    Compute integral kernels for the matrix elements in C_ij.
    Approximate the integral by assuming the invariance of integral kernel along the arc.

    """
    distance = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)

    # Notice that it is (x_j - x_i), the order is important.
    cos_alpha = (norm_x_j * (x_j - x_i) + norm_y_j * (y_j - y_i)) / distance
    z = n * k * distance
    d_gaussian = ((1j * n * k * cos_alpha) / 4) * hankel1(1, z)
    result = 2 * d_gaussian * arc_length_j

    return result


def B_ll(arc_length_j, n, k):
    """
    Compute diagonal element of B for r\'->r
    :param n: refractive index for subdomain
    :param arc_length_j: element length
    :return: Diagonal element of B
    """
    B_ll = (arc_length_j / pi) * (1 - np.log((n * k * arc_length_j) / 4) + (1j * pi / 2) - euler_gamma)
    return B_ll


def C_ll(arc_length_j, kappa, sign):
    """
    Compute diagonal element of B for r'->r.
    :param sign: inside or outside
    :param arc_length_j: element length.
    :param kappa: curvature for element.
    :return: Diagonal element of B
    """
    C_ll = -sign + kappa * arc_length_j / (2 * pi)
    return C_ll


def compute_mat_approx(n_element, cavity_size, k):
    """
    Compute the BEM matrix.
    :return: BEM matrix and the determinant of it.
    """
    Domain = domain.SingleCavityCircle(cavity_size, n_element)
    element_data = Domain.element_info
    n1 = 3.3
    n2 = 1

    x_values = element_data[:, 1]
    y_values = element_data[:, 2]

    norm_x_values = element_data[:, 3]
    norm_y_values = element_data[:, 4]
    element_curvature = element_data[:, 7]
    arc_length_values = element_data[:, 8]

    # [xi, yi, xj, yj, norm_x_j, norm_y_j, s_j, kappa_j]
    element_data_for_compute = np.zeros((n_element, n_element, 8), dtype=complex)

    # set first two column as (xi, yi) for fixed point
    # (x_ij, y_ij) = (x_i, y_i)
    element_data_for_compute[:, :, 0] = np.repeat(x_values[:, np.newaxis], n_element, axis=1)
    element_data_for_compute[:, :, 1] = np.repeat(y_values[:, np.newaxis], n_element, axis=1)

    # (x_ij, y_ij) = (x_j, y_j)
    element_data_for_compute[:, :, 2] = np.tile(x_values, (n_element, 1))
    element_data_for_compute[:, :, 3] = np.tile(y_values, (n_element, 1))
    # element_data_for_compute[:, :, 4] = np.tile(norm_x_values, (n_element, 1))
    # element_data_for_compute[:, :, 5] = np.tile(norm_y_values, (n_element, 1))

    # (norm_x_ij, norm_y_ij) = (norm_x_j, norm_y_j)
    element_data_for_compute[:, :, 4] = np.tile(norm_x_values, (n_element, 1))
    element_data_for_compute[:, :, 5] = np.tile(norm_y_values, (n_element, 1))

    # s_ij = s_j, kappa_ij = kappa_j
    element_data_for_compute[:, :, 6] = np.tile(arc_length_values, (n_element, 1))
    element_data_for_compute[:, :, 7] = element_curvature

    x_i = element_data_for_compute[:, :, 0]
    y_i = element_data_for_compute[:, :, 1]

    x_j = element_data_for_compute[:, :, 2]
    y_j = element_data_for_compute[:, :, 3]
    norm_x_j = element_data_for_compute[:, :, 4]
    norm_y_j = element_data_for_compute[:, :, 5]
    arc_length = element_data_for_compute[:, :, 6]
    kappa = element_data_for_compute[:, :, 7]

    # extract diagonal elements
    indices = np.arange(n_element)
    diagonal_element = element_data_for_compute[indices, indices, :]

    # compute B1 C1 for domain1 (inside)
    # instead of using if to check i=j, compute all element first and then fill with diagonal element is more efficient.
    B1 = B_si_sj_approx(x_i, y_i, x_j, y_j, norm_x_j, norm_y_j, arc_length, n1, k)
    C1 = C_si_sj_approx(x_i, y_i, x_j, y_j, norm_x_j, norm_y_j, arc_length, n1, k)
    # B1 = B_si_sj_approx(x_j, y_j, x_i, y_i, norm_x_i, norm_y_i, arc_length, n1, k)
    # C1 = C_si_sj_approx(x_j, y_j, x_i, y_i, norm_x_i, norm_y_i, arc_length, n1, k)

    diagonal_B1 = B_ll(diagonal_element[:, 6], n1, k)
    diagonal_C1 = C_ll(diagonal_element[:, 6], diagonal_element[:, 7], 1)

    np.fill_diagonal(B1, diagonal_B1)
    np.fill_diagonal(C1, diagonal_C1)


    # compute B2, C2
    B2 = B_si_sj_approx(x_i, y_i, x_j, y_j, norm_x_j, norm_y_j, arc_length, n2, k)
    C2 = C_si_sj_approx(x_i, y_i, x_j, y_j, norm_x_j, norm_y_j, arc_length, n2, k)
    # B2 = B_si_sj_approx(x_j, y_j, x_i, y_i, norm_x_i, norm_y_i, arc_length, n2, k)
    # C2 = C_si_sj_approx(x_j, y_j, x_i, y_i, norm_x_i, norm_y_i, arc_length, n2, k)
    # print(B1)
    diagonal_B2 = B_ll(diagonal_element[:, 6], n2, k)
    diagonal_C2 = C_ll(diagonal_element[:, 6], diagonal_element[:, 7], -1)

    np.fill_diagonal(B2, diagonal_B2)
    np.fill_diagonal(C2, diagonal_C2)

    matrix_BIE_k = np.block([[B1, C1], [B2, C2]])
    determinant = np.linalg.det(matrix_BIE_k)
    ratio = (2 * pi / (n1 * np.real(k))) / arc_length[0, 0]
    # print(ratio)
    # print(C1 - C1.T)
    # print(matrix_BIE_k)

    return matrix_BIE_k, determinant


def compute_mat_1by1(n_element, radius, k):
    Domain = domain.SingleCavityCircle(radius, n_element)
    element_data = Domain.element_info
    n1 = 3.3
    n2 = 1

    x_values = element_data[:, 1]
    y_values = element_data[:, 2]

    norm_x_values = element_data[:, 3]
    norm_y_values = element_data[:, 4]
    element_curvature = element_data[:, 7]
    arc_length_values = element_data[:, 8]

    B1 = np.zeros((n_element, n_element), dtype=complex)
    C1 = np.zeros((n_element, n_element), dtype=complex)
    B2 = np.zeros((n_element, n_element), dtype=complex)
    C2 = np.zeros((n_element, n_element), dtype=complex)

    for i in range(n_element):
        for j in range(n_element):
            if i == j:
                B1[i, j] = B_ll(arc_length_values[j], n1, k)
                C1[i, j] = -C_ll(arc_length_values[j], element_curvature[j], 1)
            else:
                B1[i, j] = B_si_sj_approx(x_values[i], y_values[i], x_values[j], y_values[j],
                                          norm_x_values[j], norm_y_values[j], arc_length_values[j], n1, k)
                C1[i, j] = C_si_sj_approx(x_values[i], y_values[i], x_values[j], y_values[j],
                                          norm_x_values[j], norm_y_values[j], arc_length_values[j], n1, k)

    for i in range(n_element):
        for j in range(n_element):
            if i == j:
                B2[i, j] = B_ll(arc_length_values[j], n2, k)
                C2[i, j] = C_ll(arc_length_values[j], element_curvature[j], -1)
            else:
                B2[i, j] = B_si_sj_approx(x_values[i], y_values[i], x_values[j], y_values[j],
                                          norm_x_values[j], norm_y_values[j], arc_length_values[j], n2, k)
                C2[i, j] = C_si_sj_approx(x_values[i], y_values[i], x_values[j], y_values[j],
                                          norm_x_values[j], norm_y_values[j], arc_length_values[j], n2, k)

    matrix_BIE_k = np.block([[B1, C1], [B2, C2]])
    determinant = np.linalg.det(matrix_BIE_k)

    return matrix_BIE_k, determinant


def compute_results_approx():
    """
    Compute the determinant for different k and save the absolute value of them in a 2d array.
    :return:
    """
    flag = 1
    k_real_values = np.linspace(0.8, 1.0, 801)
    k_imag_values = np.linspace(-0.02, 0, 80)
    num_element = 120
    radius = 10
    mat_result = np.zeros((len(k_real_values), len(k_imag_values), 3))
    time_start = time.time()
    time_sum = 0
    total = len(k_real_values)*len(k_imag_values)
    for i in range(len(k_real_values)):
        for j in range(len(k_imag_values)):
            k = k_real_values[i] + 1j * k_imag_values[j]
            matrix, determinant = compute_mat_approx(num_element, radius, k)
            mat_result[i][j][0] = k_real_values[i]
            mat_result[i][j][1] = k_imag_values[j]
            mat_result[i][j][2] = np.abs(determinant)
            elapsed_time = time.time() - time_start
            time_sum += elapsed_time

            average_time = elapsed_time / flag
            time_remaining = (total - flag) * average_time

            if flag % 10 == 0:
                print(f"{flag}/{total} jobs is finished. \n"
                      f"Used {elapsed_time:.2f} seconds.\n"
                      f"Remaining {time_remaining:.2f} seconds.")
            flag += 1

    np.save('data_try_det_0906_approx_test1.npy', mat_result)
    # print(mat_result)


def visualize_det():
    data = np.load('data_try_det_0906_ .npy')
    X = data[:, :, 0]
    Y = data[:, :, 1]
    Z = data[:, :, 2]

    plt.pcolor(X, Y, np.log(Z))
    plt.colorbar()

    plt.show()


def find_mini():
    data = np.load('data_try_det_0905_approx_test8.npy')

    k_real = data[:, :, 0]
    k_imag = data[:, :, 1]
    det_values = data[:, :, 2]


    local_min = minimum_filter(det_values, size=3)

    local_minima = (det_values == local_min)
    minima_coords = np.argwhere(local_minima)

    result = []
    for coord in minima_coords:
        x, y = coord
        result.append((k_real[x, y], k_imag[x, y], det_values[x, y]))
    for locmin in result:
        print(locmin)


def test():
    k = 0.8283-0.0089*1j
    # k = 0.82830078125 - 0.0089255859375 * 1j
    # det_n = []
    n_element = 120
    # for n in range(50, 200):
    #     n_element.append(n)
    #     det_n.append(np.abs(compute_mat_approx(n, 10, k)))
    #
    # plt.plot(n_element, det_n)
    # plt.yscale('log')
    #
    # plt.show()
    mat, det_vectorize = compute_mat_approx(n_element, 10, k)
    lu, piv = lu_factor(mat)
    lndet = np.sum(np.log(np.abs(np.diag(lu))))

    beta = int(np.log(np.abs(lndet)) / np.log(10))
    # print(beta)
    # print(lndet)
    lndet = lndet * 10 ** -beta
    det_value = np.exp(lndet)
    # det_quad = integrator.compute_matrix(n_element, 10, k)
    # det_1by1 = compute_mat_1by1(n_element, 10, k)[1]
    print(np.abs(det_vectorize))
    # print(det_quad)
    mat2, det2 = compute_mat_approx(120, 10, k)
    # print(np.abs(det2))


def test_det_fortran():
    # Initialize variables
    nbe = 30  # Replace with the actual value
    m = nbe * 2
    n = nbe * 2
    lda = nbe * 2
    k = 0.8383 - 0.0089*1j

    # Initialize T (replace with the actual matrix)
    T = compute_mat_approx(120, 10, k)[0]
    print(T)

    # LU Decomposition
    try:
        lu, piv = lu_factor(T)
        info = 0
    except np.linalg.LinAlgError:
        print("Error in LU decomposition.")
        info = 1

    # Check if LU decomposition was successful
    if info == 0:
        # Calculate the determinant using LU decomposition
        lndet = np.sum(np.log(np.abs(np.diag(lu))))

        # Calculate beta
        beta = int(np.log10(np.abs(lndet)))

        # Output results
        print(f"# beta = {beta}; lndet = {lndet}")
    det = np.exp(lndet / 10)
    # print(det)

if __name__ == '__main__':
    # compute_results_approx()
    # visualize_det()
    # find_mini()  # [(0.8334, -0.0002)]

    test()  # 7.084260440445197e-07
    # (3.682741463454434e-83-8.673392258202733e-83j)
    # test_det_fortran()
