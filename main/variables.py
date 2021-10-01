# import numpy as np
import cupy as cp


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        self.arr_spectral = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, norm='forward'))

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr_spectral), norm='forward'))

    def integrate(self, grid):
        arr_add = cp.append(self.arr_nodal, self.arr_nodal[0])
        return trapz(arr_add, grid.x.dx)

    def integrate_energy(self, grid):
        arr = 0.5 * self.arr_nodal ** 2.0
        arr_add = cp.append(arr, arr[0])
        return trapz(arr_add, grid.x.dx)


class Distribution:
    def __init__(self, resolutions, order):
        self.x_res, self.u_res, self.v_res = resolutions
        self.order = order

        # arrays
        self.arr, self.arr_nodal = None, None
        self.zero_moment = SpaceScalar(resolution=resolutions[0])
        self.second_moment = SpaceScalar(resolution=resolutions[0])

    def compute_zero_moment(self, grid):
        self.inverse_fourier_transform()
        self.zero_moment.arr_nodal = grid.u.zero_moment(
            function=grid.v.zero_moment(function=self.arr_nodal,
                                        idx=[3, 4]),
            idx=[1, 2])
        self.zero_moment.fourier_transform()

    def total_thermal_energy(self, grid):
        self.inverse_fourier_transform()
        # self.second_moment.arr_nodal = grid.u.second_moment(
        #     function=grid.v.second_moment(function=self.arr_nodal,
        #                                   dim=2,
        #                                   idx=[3, 4]),
        #     dim=1,
        #     idx=[1, 2])
        integrand = grid.v_mag_sq[None, :, :, :, :] * self.arr_nodal
        self.second_moment.arr_nodal = grid.u.zero_moment(
            function=grid.v.zero_moment(function=integrand,
                                        idx=[3, 4]),
            idx=[1, 2])
        return 0.5 * self.second_moment.integrate(grid=grid)

    def total_density(self, grid):
        self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return self.zero_moment.integrate(grid=grid)

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.u_res * self.order, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.u_res * self.order, self.v_res * self.order)

    def initialize(self, grid):
        ix, iu, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        maxwellian = cp.tensordot(ix, cp.tensordot(grid.u.compute_maxwellian(thermal_velocity=1.0,
                                                                             drift_velocity=0.0),
                                                   grid.v.compute_maxwellian(thermal_velocity=1.0,
                                                                             drift_velocity=0.0),
                                                   axes=0),
                                  axes=0)

        # compute perturbation
        # perturbation = cp.imag(grid.eigenfunction(thermal_velocity=1,
        #                                           drift_velocity=[2, -2],
        #                                           eigenvalue=1.20474886j))
        perturbation = cp.multiply(cp.sin(grid.x.fundamental * grid.x.device_arr)[:, None, None, None, None], maxwellian)
        self.arr_nodal = maxwellian + 0.01 * perturbation

    def fourier_transform(self):
        self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)

    def inverse_fourier_transform(self):
        self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0
