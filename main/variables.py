# import numpy as np
import cupy as cp


class SpaceScalar:
    def __init__(self, resolution):
        self.res = resolution
        self.arr_nodal, self.arr_spectral = None, None

    def fourier_transform(self):
        # self.arr_spectral = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, norm='forward'))
        self.arr_spectral = cp.fft.rfft(self.arr_nodal, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr_spectral), norm='forward'))
        self.arr_nodal = cp.fft.irfft(self.arr_spectral, norm='forward')

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
        # self.inverse_fourier_transform()
        self.zero_moment.arr_spectral = grid.u.zero_moment(
            function=grid.v.zero_moment(function=self.arr,
                                        idx=[3, 4]),
            idx=[1, 2])
        self.zero_moment.inverse_fourier_transform()

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
        # self.inverse_fourier_transform()
        self.compute_zero_moment(grid=grid)
        return self.zero_moment.integrate(grid=grid)

    def grid_flatten(self):
        return self.arr_nodal.reshape(self.x_res, self.u_res * self.order, self.v_res * self.order)

    def spectral_flatten(self):
        return self.arr.reshape(self.arr.shape[0], self.u_res * self.order, self.v_res * self.order)

    def initialize(self, grid):
        ix, iu, iv = cp.ones_like(grid.x.device_arr), cp.ones_like(grid.u.device_arr), cp.ones_like(grid.v.device_arr)
        # maxwellian = cp.tensordot(ix, cp.tensordot(grid.u.compute_maxwellian(thermal_velocity=1.0,
        #                                                                      drift_velocity=0.0),
        #                                            grid.v.compute_maxwellian(thermal_velocity=1.0,
        #                                                                      drift_velocity=0.0),
        #                                            axes=0),
        #                           axes=0)
        ring_distribution = cp.tensordot(ix, grid.ring_distribution(thermal_velocity=1.0,
                                                                    ring_parameter=2.0 * cp.pi),
                                         axes=0)

        # compute perturbation
        # Examples: L = 2pi, first mode:  1.16387241 + 0j
        #           L = pi, first mode: 1.03859465
        #           L = pi, second mode: 2.05498248
        #           L = pi, third mode: 3.04616847
        perturbation = grid.eigenfunction(thermal_velocity=1,
                                          ring_parameter=2.0 * cp.pi,
                                          eigenvalue=-3.48694202e-01j,
                                          parity=False)

        # perturbation = cp.multiply(cp.sin(grid.x.fundamental *
        # grid.x.device_arr)[:, None, None, None, None], maxwellian)

        self.arr_nodal = ring_distribution + 1.0e-3 * perturbation

    def fourier_transform(self):
        # self.arr = cp.fft.fftshift(cp.fft.fft(self.arr_nodal, axis=0, norm='forward'), axes=0)
        self.arr = cp.fft.rfft(self.arr_nodal, axis=0, norm='forward')

    def inverse_fourier_transform(self):
        # self.arr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.arr, axes=0), norm='forward', axis=0))
        self.arr_nodal = cp.fft.irfft(self.arr, axis=0, norm='forward')


def trapz(y, dx):
    """ Custom trapz routine using cupy """
    return cp.sum(y[:-1] + y[1:]) * dx / 2.0
