from __future__ import division
import numpy as np
from fractions import Fraction


def etok(e_mev):
    """
    Converts meV to wavevector Angstrom^-1.
    :param e_mev: energy in meV.
    :return: Wavevector in Angstrom^-1.
    """
    return np.sqrt(e_mev) * 0.6947


def ktoe(k_inv_angstrom):
    """
    Does the opposite or etok.
    :param k_inv_angstrom: A-1.
    :return: meV.
    """
    return (k_inv_angstrom / 0.6947) ** 2


class UBMatrix(object):
    """
    UB-matrix handling object. Can be compared to each other using == operator to check if all settings are identical.
    Example: Hexagonal crystal with a = 4.05A, c = 11.05A, aligned on hkl1 = [1 0 0] and hkl2 = [0 1 0], we want x axis
    to be [1 1 0] and y axis to be [1 -1 0]
    UBMatrix([4.05, 4.05, 11.05, 90, 90, 120], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0])
    """
    def __init__(self, latparam, hkl1, hkl2, plot_x=None, plot_y=None, a3_add=0.0, a4_add=0.0):
        """
        Create a UBMatrix object.
        :param latparam: Lattice parameters in list form [a, b, c, alpha, beta, gamma] in Angstroms and degrees.
        :param hkl1: Scattering plane HKL1 [h, k, l].
        :param hkl2: Scattering plane HKL2 [h, k, l].
        :param plot_x: x-axis of 2D-plots, [h, k, l]. Can be omitted if plot_x == hkl1 and plot_y == hkl2.
        :param plot_y: y-axis of 2D-plots, [h, k, l]. Can be omitted if plot_x == hkl1 and plot_y == hkl2.
        """
        self._latparam = latparam
        self._hkl1 = hkl1
        self._hkl2 = hkl2
        if plot_x is None or plot_y is None:
            self._plot_x = self._hkl1
            self._plot_y_nominal = self._hkl2
            self._plot_y = None
        else:
            self._plot_x = plot_x
            self._plot_y_nominal = plot_y
            self._plot_y = None
        self._theta = None
        self.shear_coeff = None

        self.conversion_matrices = None
        self.figure_aspect = None
        self.a3_add = a3_add
        self.a4_add = a4_add
        self.update_conversion_matrices()

    @property
    def latparam(self):
        return self._latparam

    @property
    def hkl1(self):
        return self._hkl1

    @property
    def hkl2(self):
        return self._hkl2

    @property
    def plot_x(self):
        return self._plot_x

    @property
    def plot_y_nominal(self):
        return self._plot_y_nominal

    @property
    def plot_y_actual(self):
        return self._plot_y

    @property
    def is_orthogonal(self):
        return np.isclose(self._theta, np.pi / 2)

    @property
    def theta(self):
        return self._theta

    def update_conversion_matrices(self):
        self.conversion_matrices = dict()
        self.conversion_matrices['ll'] = np.diag([1, 1, 1])
        self.conversion_matrices['bl'] = self._calculate_bl()
        self.conversion_matrices['rl'] = self._calculate_rl()
        self.conversion_matrices['sl'] = self._calculate_sl()
        self._update_plot_axes()
        self.conversion_matrices['pl'] = self._calculate_pl()
        self._calculate_aspect()

    def _update_plot_axes(self):
        norm = np.linalg.norm
        plot_x_s = self.convert(self._plot_x, 'rs')
        plot_nominal_y_s = self.convert(self._plot_y_nominal, 'rs')
        if np.isclose(np.dot(plot_x_s, plot_nominal_y_s), 0):
            self._plot_y = self._plot_y_nominal[:]
            self._theta = np.pi / 2
            self.shear_coeff = 0
        else:
            self._theta = np.arccos(np.dot(plot_x_s, plot_nominal_y_s) / (norm(plot_x_s) * norm(plot_nominal_y_s)))
            plot_y_s = rotate_around_z(plot_x_s, np.pi/2) / norm(plot_x_s) * norm(plot_nominal_y_s) * np.sin(self.theta)
            plot_y_r = self.convert(plot_y_s, 'sr')
            self._plot_y = plot_y_r
            self.shear_coeff = norm(plot_nominal_y_s) * np.cos(self.theta) / norm(plot_x_s)

    def _calculate_bl(self):
        lattice_parameters = self._latparam
        a, b, c, alpha_deg, beta_deg, gamma_deg = lattice_parameters

        cos_alpha = np.cos(np.radians(alpha_deg))
        cos_beta = np.cos(np.radians(beta_deg))
        cos_gamma = np.cos(np.radians(gamma_deg))
        sin_gamma = np.sin(np.radians(gamma_deg))
        tan_gamma = np.tan(np.radians(gamma_deg))

        ax = a
        a_l = np.array([[ax], [0], [0]])

        bx = b * cos_gamma
        by = b * sin_gamma
        b_l = np.array([[bx], [by], [0]])

        cx = c * cos_beta
        cy = c * (cos_alpha / sin_gamma - cos_beta / tan_gamma)
        cz = c * np.sqrt(1 - (cx ** 2 + cy ** 2) / c ** 2)
        c_l = np.array([[cx], [cy], [cz]])

        b_in_l = np.concatenate((a_l, b_l, c_l), axis=1)

        return b_in_l

    def _calculate_rl(self):
        r_in_l = (2 * np.pi * np.linalg.inv(self.conversion_matrices['bl'])).T
        return r_in_l

    def _calculate_sl(self):
        hkl1_l = np.dot(self.conversion_matrices['rl'], self._hkl1)
        hkl2_l = np.dot(self.conversion_matrices['rl'], self._hkl2)

        hkl1_cross_hkl2 = np.cross(hkl1_l, hkl2_l)
        sz_in_l = hkl1_cross_hkl2 / np.linalg.norm(hkl1_cross_hkl2)
        sx_in_l = hkl1_l / np.linalg.norm(hkl1_l)
        sy_in_l = np.cross(sz_in_l, sx_in_l)

        s_in_l = np.array([sx_in_l, sy_in_l, sz_in_l]).T
        return s_in_l

    def _calculate_pl(self):
        px_in_l = np.dot(self.conversion_matrices['rl'], self._plot_x)
        py_in_l = np.dot(self.conversion_matrices['rl'], self._plot_y)
        pz_in_l = np.cross(px_in_l, py_in_l)  # for sake of completeness

        p_in_l = np.array([px_in_l, py_in_l, pz_in_l]).T
        return p_in_l

    def _calculate_aspect(self):
        plot_x_r = self._plot_x
        plot_y_r = self._plot_y
        plot_x_s = self.convert(plot_x_r, 'rs')
        plot_y_s = self.convert(plot_y_r, 'rs')
        len_x = np.linalg.norm(plot_x_s)
        len_y = np.linalg.norm(plot_y_s)
        self.figure_aspect = len_y / len_x

    def angle_to_q(self, ki, kf, a3, a4, system='s'):
        return angle_to_q(ki, kf, a3, a4, self.a3_add, self.a4_add, system=system, ub_matrix=self)

    def find_spurion(self, q, ki, kf, type='a', in_system='r', out_system='r'):
        return find_spurion(q, ki, kf, self, type, in_system, out_system)

    def convert(self, vectors, sys, axis=1):
        """
        Convert between coordinate systems in UB-matrix
        r-Reciprocal, s-Sample table, p-Plot
        :param vectors: xyz or N*3 array or 3*N array.
        :param sys: String representation of source and destination system. 'rs' converts from r to s system.
        :param axis: Works along which axis.
        :return: Vector or ndarray of same dimension as input.
        """
        vectors = np.asarray(vectors)
        try:
            conversion_matrix = self.conversion_matrices['sys']
        except KeyError:
            source_system = sys[0]
            target_system = sys[1]
            source_to_l_matrix = self.conversion_matrices[source_system + 'l']
            target_to_l_matrix = self.conversion_matrices[target_system + 'l']
            conversion_matrix = np.dot(np.linalg.inv(target_to_l_matrix), source_to_l_matrix)
            self.conversion_matrices[sys] = conversion_matrix
            self.conversion_matrices[sys[::-1]] = np.linalg.inv(conversion_matrix)
        finally:
            pass
        if axis == 1:
            return np.dot(conversion_matrix, vectors)
        elif axis == 0:
            return np.dot(conversion_matrix, vectors.T).T
        else:
            raise ValueError('invalid axis for ub-matrix vectors')

    def __copy__(self):
        return UBMatrix(self._latparam, self.hkl1, self.hkl2, self.plot_x, self.plot_y_nominal)

    def copy(self):
        return self.__copy__()

    def __eq__(self, other):
        """
        Check if two UBMatrix objects are the same.
        :param other: UBMatrix to be checked against
        :return: bool
        """
        # type: (UBMatrix) -> bool
        if not isinstance(other, UBMatrix):
            raise TypeError('UBMatrix can only be checked for equality with another UBMatrix.')
        latparam_eq = np.all(self.latparam == other.latparam)
        hkl_eq = np.all(self.hkl1 == other.hkl1) and np.all(self.hkl2 == other.hkl2)
        plot_eq = np.all(self.plot_x == other.plot_x) and np.all(self.plot_y_nominal == other.plot_y_nominal)

        if latparam_eq and hkl_eq and plot_eq:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


def rotate_around_z(vectors, angles, axis=1):
    """
    Rotates given vectors around third axis.
    :param vectors: numpy.array, vectors to be rotated
    :param angles: angles in radians.
    :param axis: 1 for column vectors, 0 for row vectors.
    :return: rotated vectors.
    """
    try:
        single_vector = False
        if axis == 1:
            x, y, z = vectors[0, :], vectors[1, :], vectors[2, :]
        elif axis == 0:
            x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
        else:
            raise ValueError('rotation axis should be either 1 or 0, provided: %d' % axis)
    except (IndexError, TypeError):
        single_vector = True
        x, y, z = vectors[0], vectors[1], vectors[2]
    cosines = np.cos(angles)
    sines = np.sin(angles)
    results = np.vstack([x * cosines - y * sines, x * sines + y * cosines, z])
    if single_vector:
        return results.ravel()
    if axis == 1:
        return results
    else:
        return results.T


def guess_axes_labels(hkl1, hkl2):
    labels = ('H', 'K', 'L')
    hkl1_nonzero = [not np.isclose(x, 0) for x in hkl1]
    hkl2_nonzero = [not np.isclose(x, 0) for x in hkl2]

    if hkl1_nonzero.count(True) == 0 or hkl2_nonzero.count(True) == 0:
        raise ValueError('guess axis labels: axis supplied is all zeroes.')
    if hkl2_nonzero.count(True) == 1:  # axis with only one component gets precedence
        symbol_y = labels[hkl2_nonzero.index(True)]
        if labels[hkl1_nonzero.index(True)] != symbol_y:  # check if same symbol used twice
            symbol_x = labels[hkl1_nonzero.index(True)]
        else:
            hkl1_nonzero[hkl1_nonzero.index(True)] = False
            try:
                symbol_x = labels[hkl1_nonzero.index(True)]
            except ValueError:
                raise ValueError('guess axis labels: supplied axes are co-linear.')
    else:
        symbol_x = labels[hkl1_nonzero.index(True)]
        symbol_y = labels[hkl2_nonzero.index(True)]
        if symbol_x == symbol_y:
            hkl2_nonzero[hkl1_nonzero.index(True)] = False
            symbol_y = labels[hkl2_nonzero.index(True)]

    hkl1_strings = make_label(hkl1, symbol_x)
    hkl2_strings = make_label(hkl2, symbol_y)
    return hkl1_strings, hkl2_strings


def make_label(hkl, symbol, return_list=False):
    hkl_fractions = [Fraction(x).limit_denominator(10) for x in hkl]
    use_absolute = [np.abs(x - y) > 0.01 for x, y in zip(hkl_fractions, hkl)]
    label_str_list = []

    for i in range(3):
        if hkl[i] == 0:
            label_str_list.append('0')
        elif hkl[i] == 1:
            label_str_list.append(symbol)
        elif hkl[i] == -1:
            label_str_list.append('-' + symbol)
        elif use_absolute[i]:
            label_str_list.append('%.2f' % hkl[i] + symbol)
        else:
            label_str_list.append(str(hkl_fractions[i]) + symbol)

    if not return_list:
        label_string = '[' + ' '.join(label_str_list) + ']'
        return label_string

    return label_str_list


def find_triangle_angles(a, b, c):
    """
    calculates triangle angles from edge lengths.
    :param a: length of a
    :param b: length of b
    :param c: length of c
    :return: alpha, beta, gamma angles in radians
    """
    aa = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    bb = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cc = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    if abs(aa) > 1 or abs(bb) > 1 or abs(cc) > 1:
        raise ValueError('triangle does not close.')
    alpha = np.arccos(aa)
    beta = np.arccos(bb)
    gamma = np.arccos(cc)

    return alpha, beta, gamma


def angle_to_q(ki, kf, a3, a4, a3_add=0.0, a4_add=0.0, system='s', ub_matrix=None):
    """
    # type: (float, float, ...) -> ...
    :param ki: ki, in angstroms^-1
    :param kf: kf, in angstroms^-1
    :param a3: np.ndarray or list or float of A3 angles, in degrees
    :param a4: A4 angles, in degrees.
    :param a3_add: value to be ADDED to A3, in degrees.
    :param a4_add: value to be ADDED to A4, in degrees.
    :param system: Output system, ('s', 'r', 'p')
    :param ub_matrix: UBMatrix object, required if system is not 's'.
    :return: q vectors in S system, 3xN array if input is in iterable form.
    """
    try:
        length = min(len(a3), len(a4))
    except TypeError:
        length = 1
    a3 = np.asarray(a3).reshape(1, -1) + a3_add
    a4 = np.asarray(a4).reshape(1, -1) + a4_add
    ones = np.ones([1, length])
    zeros = np.zeros([1, length])
    initial_vectors = np.vstack([-ones, zeros, zeros])
    try:
        len_ki = len(ki)
        if len_ki == length:
            ki_in_s = np.array(ki) * initial_vectors
        else:
            ki_in_s = ki[0] * initial_vectors
    except TypeError:
        ki_in_s = ki * initial_vectors
    kf_in_s = kf * rotate_around_z(initial_vectors, np.radians(a4))
    q_in_s = rotate_around_z(kf_in_s - ki_in_s, - np.radians(a3))
    if system == 's':
        return q_in_s
    else:
        if UBMatrix is None:
            raise TypeError('UBMatrix required for angle to Q if output system is not sample system.')
        return ub_matrix.convert(q_in_s, 's' + system)


def v1_to_v2(v1, v2):
    norm = np.linalg.norm
    cosine = np.dot(v1, v2) / (norm(v1) * norm(v2))
    acos = np.arccos(cosine)
    if np.cross(v1, v2)[2] >= 0:
        return acos
    else:
        return -acos


def find_a3_a4(q, ki, kf, ub_matrix, system='r', sense=1):
    # type: ((np.ndarray, list), float, float, UBMatrix, str, int) -> (float, float)
    """
    Finds A3 and A4 angle for given vector.
    :param q: 3-element Q vector. Does not accept a list of vectors.
    :param ki: ki
    :param kf: kf
    :param ub_matrix: UBMatrix object.
    :param system: In which system is the vector given. r for reciprocal vectors.
    :param sense: Scattering sense, +1 for CCW scattering
    :return: (a3, a4) in degrees
    """
    q_in_s = ub_matrix.convert(q, system + 's')
    if not np.isclose(q_in_s[2], 0):
        raise ValueError('Q not in scattering plane.')
    alpha, beta, gamma = find_triangle_angles(np.linalg.norm(q_in_s), ki, kf)

    a4_rad = alpha * sense
    ki_in_s = rotate_around_z(q_in_s, gamma * sense + np.pi) / np.linalg.norm(q_in_s) * ki
    ki_zero_s = ub_matrix.convert(np.asarray(ub_matrix.hkl1) * -1, 'rs')
    a3_rad = -1 * v1_to_v2(ki_zero_s, ki_in_s)
    return a3_rad * 180 / np.pi, a4_rad * 180 / np.pi


def find_spurion(q, ki, kf, ub_matrix, type='a', in_system='r', out_system='r'):
    """
    Find nominal position of spurion from accidental Bragg scattering for give parameters <Shirane, 2002, p.148>
    :param q: length-3 vector of Bragg reflection
    :param ki: nominal incoming wavevector , 1/Angstrom.
    :param kf: nominal final wavevector
    :param ub_matrix: UBMatrix object.
    :param type: 'a' = Typ A, 'm' = Typ M
    :param in_system: Input vector coordinate system, ('r', 's', 'p')
    :param out_system: Output vector coordinate system, ('r', 's', 'p')
    :return: length-3 vector of spurion position.
    """
    # type: ((np.ndarray, list), float, float, UBMatrix, str, str, str) -> np.ndarray
    qr = ub_matrix.convert(q, in_system + 'r')

    if type.lower() == 'a':
        a3, a4 = find_a3_a4(qr, ki, ki, ub_matrix)
    elif type.lower() == 'm':
        a3, a4 = find_a3_a4(qr, kf, kf, ub_matrix)
    else:
        raise ValueError('Spurion type should be either a (Typ A) or m (Typ M).')
    spurion_s = angle_to_q(ki, kf, a3, a4)
    return ub_matrix.convert(spurion_s, 's' + out_system).ravel()
