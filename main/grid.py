import numpy as np
import cupy as cp
import basis as b


class SpaceGrid:
    """ In this scheme, the spatial grid is uniform and transforms are accomplished by DFT """

    def __init__(self, low, high, elements):
        # grid limits and elements
        self.low, self.high = low, high
        self.elements = elements

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # element Jacobian
        self.J = 2.0 / self.dx

        # arrays
        self.arr, self.device_arr = None, None
        self.create_grid()

        # spectral properties
        self.modes = elements // 2.0  # Nyquist frequency
        self.fundamental = 2.0 * np.pi / self.length
        self.wavenumbers = self.fundamental * np.arange(-self.modes, self.modes)
        self.device_wavenumbers = cp.array(self.wavenumbers)
        self.zero_idx = int(self.modes)
        # self.two_thirds_low = int((1 * self.modes)//3 + 1)
        # self.two_thirds_high = self.wavenumbers.shape[0] - self.two_thirds_low
        self.pad_width = int((1 * self.modes) // 3 + 1)
        # print(self.two_thirds_low)
        # print(self.two_thirds_high)

    def create_grid(self):
        """ Build evenly spaced grid, assumed periodic """
        self.arr = np.linspace(self.low, self.high - self.dx, num=self.elements)
        self.device_arr = cp.asarray(self.arr)


class VelocityGrid:
    """ In this experiment, the velocity grid is an LGL quadrature grid """

    def __init__(self, low, high, elements, order):
        self.low, self.high = low, high
        self.elements, self.order = elements, order
        self.local_basis = b.LGLBasis1D(order=self.order)
        # self.local_basis = b.GLBasis1D(order=self.order)

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.elements

        # jacobian
        self.J = 2.0 / self.dx

        # global quad weights
        self.global_quads = cp.tensordot(cp.ones(elements),
                                         cp.asarray(self.local_basis.weights), axes=0)

        # arrays
        self.arr, self.device_arr = None, None
        self.mid_points = None
        self.create_grid()

        # global translation matrix
        mid_identity = np.tensordot(self.mid_points, np.eye(self.local_basis.order), axes=0)
        self.translation_matrix = cp.asarray(mid_identity + self.local_basis.translation_matrix / self.J)

    def create_grid(self):
        """ Build global grid """
        # translate to [0, 1]
        nodes_iso = (np.array(self.local_basis.nodes) + 1) / 2
        # element left boundaries (including ghost elements)
        xl = np.linspace(self.low, self.high - self.dx, num=self.elements)
        # construct coordinates
        self.arr = np.zeros((self.elements, self.order))
        for i in range(self.elements):
            self.arr[i, :] = xl[i] + self.dx * np.array(nodes_iso)
        # send to device
        self.device_arr = cp.asarray(self.arr)
        self.mid_points = np.array([0.5 * (self.arr[i, -1] + self.arr[i, 0]) for i in range(self.elements)])

    def zero_moment(self, function, idx):
        return cp.tensordot(self.global_quads, function, axes=([0, 1], idx)) / self.J

    def second_moment(self, function, dim, idx):
        if dim == 2:
            return cp.tensordot(self.global_quads, cp.multiply(self.device_arr[None, None, None, :, :] ** 2.0,
                                                               function),
                                axes=([0, 1], idx)) / self.J
        if dim == 1:
            return cp.tensordot(self.global_quads, cp.multiply(self.device_arr[None, :, :] ** 2.0,
                                                               function),
                                axes=([0, 1], idx)) / self.J

    def compute_maxwellian(self, thermal_velocity, drift_velocity):
        return cp.exp(-((self.device_arr - drift_velocity) /
                        thermal_velocity) ** 2.0) / (np.sqrt(np.pi) * thermal_velocity)

    def compute_maxwellian_gradient(self, thermal_velocity, drift_velocity):
        return (-2.0 * ((self.device_arr - drift_velocity) / thermal_velocity ** 2.0) *
                self.compute_maxwellian(thermal_velocity=thermal_velocity, drift_velocity=drift_velocity))


class PhaseSpace:
    """ In this experiment, PhaseSpace consists of equispaced nodes and a
    LGL tensor-product grid in truncated velocity space """

    def __init__(self, lows, highs, elements, order):
        self.x = SpaceGrid(low=lows[0], high=highs[0], elements=elements[0])
        self.u = VelocityGrid(low=lows[1], high=highs[1], elements=elements[1], order=order)
        self.v = VelocityGrid(low=lows[2], high=highs[2], elements=elements[2], order=order)

        self.v_mag_sq = self.u.device_arr[:, :, None, None] ** 2.0 + self.v.device_arr[None, None, :, :] ** 2.0

        # self.translation_matrix = cp.tensordot(self.u.translation_matrix, cp.eye(self.v.order), axes=0)
        # print(self.translation_matrix.shape)

    # def eigenfunction(self, thermal_velocity, drift_velocity, eigenvalue, beams='two-stream'):
    #     if beams == 'two-stream':
    #         df1 = self.u.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
    #                                                  drift_velocity=drift_velocity[0])
    #         df2 = self.v.compute_maxwellian_gradient(thermal_velocity=thermal_velocity,
    #                                                  drift_velocity=drift_velocity[1])
    #         df = 0.5 * (df1 + df2)
    #         v_part = cp.divide(df, self.v.device_arr - eigenvalue)
    #         return cp.tensordot(cp.exp(1j * self.x.fundamental * self.x.device_arr), v_part, axes=0)
