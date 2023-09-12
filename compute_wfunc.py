import numpy as np
import matplotlib.pyplot as plt
import domain
from compute_B_C_approx import compute_mat_approx
from scipy.special import hankel1
from scipy.linalg import eig
from numpy import pi, euler_gamma
import os


def compute_null_space(A):
    # Compute the SVD
    U, s, Vh = np.linalg.svd(A)

    # Find the index of the smallest singular value
    idx = np.argmin(s)

    # Return the right singular vector corresponding to the smallest singular value
    null_vector = Vh[idx, :].T
    print(null_vector)
    return null_vector


def find_nth_min_value(arr, n):
    # Sort the array and get the nth smallest value
    sorted_indices = np.argsort(arr)
    nth_min_value = arr[sorted_indices[n - 1]]

    # Find the index (location) of the nth smallest value in the original array
    index = np.where(arr == nth_min_value)[0]

    return index


def compute_wave_mat(n_element, cavity_size, x_min, x_max, y_min, y_max, num_x, num_y, k, save_loc='./'):
    test = 0
    r = cavity_size
    n_in = 3.3
    n_out = 1
    x_values = np.linspace(x_min, x_max, num_x)
    y_values = np.linspace(y_min, y_max, num_y)

    # compute BEM matrix to find the null vector
    matrix, determinant = compute_mat_approx(n_element, r, k)

    eigenvalues, eigenvectors = eig(matrix)

    smallest_eigenvalue_index = find_nth_min_value(np.abs(eigenvalues), 1)

    # The eigenvectors are the columns of the eigenvectors-matrix
    null_vector_approx = eigenvectors[:, smallest_eigenvalue_index]

    # values of phi on the boundary
    phi_n = null_vector_approx[n_element:]
    phi_origin_fortran = np.loadtxt('dat.phi').reshape((120, 2))
    phi_n_fortran = phi_origin_fortran[:, 0] + 1j * phi_origin_fortran[:, 1]
    # TM-mode
    d_phi_in = null_vector_approx[:n_element]
    d_phi_out = null_vector_approx[:n_element]
    d_phi_in_origin_fortran = np.loadtxt('dat.dphi').reshape((120, 2))
    d_phi_in_fortran = d_phi_in_origin_fortran[:, 0] + 1j * d_phi_in_origin_fortran[:, 1]
    # d_phi_out = d_phi_in

    null_vector_fortran = np.concatenate((d_phi_in_fortran, phi_n_fortran))

    if test == 1:
        plt.plot(range(120), np.real(null_vector_fortran[120:]), '.', linewidth=0.8)
        plt.show()
        return

    data_wave = np.zeros((num_x, num_y, 3), dtype=complex)
    Boundary = domain.SingleCavityCircle(r, n_element)

    data_boundary = Boundary.element_info
    element_x = data_boundary[:, 1]
    element_y = data_boundary[:, 2]
    element_kappa = data_boundary[:, 7]
    element_length = data_boundary[:, 8]
    element_norm_x = data_boundary[:, 3]
    element_norm_y = data_boundary[:, 4]

    count = 0
    for i in range(num_x):
        for j in range(num_y):
            count += 1
            if count % 100 == 0:
                print(f'{count} / {num_x*num_y} is finished')
            x, y = x_values[i], y_values[j]
            data_wave[i, j, 0] = x
            data_wave[i, j, 1] = y
            if x in element_x and y in element_y:
                data_wave[i, j, 2] = 0
            if Boundary.is_inside_cavity(x, y):
                nk_in = n_in * k
                for ni in range(n_element):
                    data_wave[i, j, 2] += compute_wave(nk_in, x, y, element_x[ni], element_y[ni],
                                                       element_norm_x[ni], element_norm_y[ni],
                                                       element_length[ni], element_kappa[ni],
                                                       phi_n[ni], d_phi_in[ni])

            else:
                nk_out = n_out * k
                for ni in range(n_element):
                    data_wave[i, j, 2] += compute_wave(nk_out, x, y, element_x[ni], element_y[ni],
                                                       element_norm_x[ni], element_norm_y[ni],
                                                       element_length[ni], element_kappa[ni],
                                                       phi_n[ni], d_phi_out[ni])
    path_save = os.path.join(save_loc, f'k{k.real}_{k.imag}_r{r}_n{n_element}.npy')
    np.save(path_save, data_wave)


def compute_wave(nk, x, y, element_x, element_y, element_norm_x, element_norm_y, element_length,
                 element_kappa, phi, dphi):
    # compute the wave function for one point.
    small = 1e-2  # Points close to the boundary shorter than this distance should use another method.
    result = 0

    # contribution 1 from norm derivative of wave function on the boundary.
    w1 = d_greenfunc_correct(nk, element_x, element_y, x, y, element_norm_x, element_norm_y, element_kappa, small)

    result += w1 * phi * element_length

    # contribution 2 from the wave function on the boundary.
    w2 = greenfunc_correct(nk, element_x, element_y, x, y, element_length, small)

    result -= w2 * dphi * element_length

    return result


def d_greenfunc_correct(nk, xl, yl, x, y, nx, ny, kappa, small):
    dx = xl - x
    dy = yl - y
    distance = np.sqrt(dx*dx + dy*dy)

    if distance <= small:
        w = kappa / (4 * pi)
        return w
    else:
        w = d_greenfunc(nk, xl, yl, x, y, nx, ny)
        return w


def d_greenfunc(nk, x1, y1, x2, y2, nx, ny):
    dx = x1 - x2
    dy = y1 - y2
    distance = np.sqrt(dx*dx + dy*dy)
    cos_theta = (nx * dx + ny * dy) / distance

    z = nk * distance
    result = 0.25 * 1j * nk * cos_theta * hankel1(1, z)

    return result


def greenfunc_correct(nk, xl, yl, x, y, ds, small):
    dx = xl - x
    dy = yl - y
    distance = np.sqrt(dx * dx + dy * dy)
    if distance < small:
        w = (-1 / (2*pi)) * (1 - np.log((nk * ds) / 4) + (1j * pi / 2) - euler_gamma)
        return w
    else:
        w = greenfunc(nk, xl, yl, x, y)
        return w


def greenfunc(nk, x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    z = nk * np.sqrt(dx*dx + dy*dy)
    result = -0.25 * 1j * hankel1(0, z)

    return result


def plot_wfunc():
    data = np.load('k0.8283_-0.0089_r10_n120.npy')
    print(data)
    x_values = data[:, :, 0]
    y_values = data[:, :, 1]
    intensity = np.abs(data[:, :, 2])

    plt.pcolor(np.real(x_values), np.real(y_values), intensity**2)

    plt.colorbar()

    plt.show()


def main_compute_wfunc():
    n_element = 120
    cavity_size = 10
    x_min, x_max, y_min, y_max = -15, 15, -15, 15
    num_x = 999
    num_y = 999
    k = 0.8283 - 0.0089*1j
    compute_wave_mat(n_element, cavity_size, x_min, x_max, y_min, y_max, num_x, num_y, k)


if __name__ == '__main__':
    # main_compute_wfunc()

    plot_wfunc()
