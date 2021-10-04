import numpy as np
import cupy as cp
import variables as var
import plotter as my_plt
import time as timer


def basis_product(flux, basis_arr, axis, permutation):
    return cp.transpose(cp.tensordot(flux, basis_arr,
                                     axes=([axis], [1])),
                        axes=permutation)


class DGFlux:
    def __init__(self, resolutions, order, om_pc):
        self.x_ele, self.u_res, self.v_res = resolutions
        self.x_res = int(self.x_ele // 2 + 1)
        self.order = order

        # permutations
        self.permutations = [(0, 1, 4, 2, 3),  # for contraction with u nodes
                             (0, 1, 2, 3, 4)]  # for contraction with v nodes

        # slices into the DG boundaries (list of tuples)
        self.boundary_slices = [[(slice(self.x_res), slice(self.u_res), 0, slice(self.v_res), slice(self.order)),
                                 (slice(self.x_res), slice(self.u_res), -1, slice(self.v_res), slice(self.order))],
                                [(slice(self.x_res), slice(self.u_res), slice(self.order), slice(self.v_res), 0),
                                 (slice(self.x_res), slice(self.u_res), slice(self.order), slice(self.v_res), -1)]]
        self.boundary_slices_pad = [[(slice(self.x_res),
                                      slice(self.u_res + 2), 0,
                                      slice(self.v_res), slice(self.order)),
                                     (slice(self.x_res),
                                      slice(self.u_res + 2), -1,
                                      slice(self.v_res), slice(self.order))],
                                    [(slice(self.x_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res + 2), 0),
                                     (slice(self.x_res),
                                      slice(self.u_res), slice(self.order),
                                      slice(self.v_res + 2), -1)]]
        self.flux_input_slices = [(slice(self.x_res), slice(1, self.u_res + 1), slice(self.order),
                                   slice(self.v_res), slice(self.order)),
                                  (slice(self.x_res), slice(self.u_res), slice(self.order),
                                   slice(1, self.v_res + 1), slice(self.order))]
        self.pad_slices = [(slice(self.x_res), slice(1, self.u_res + 1),
                            slice(self.v_res), slice(self.order)),
                           (slice(self.x_res), slice(self.u_res), slice(self.order),
                            slice(1, self.v_res + 1))]
        self.num_flux_sizes = [(self.x_res, self.u_res, 2, self.v_res, self.order),
                               (self.x_res, self.u_res, self.order, self.v_res, 2)]
        self.padded_flux_sizes = [(self.x_res, self.u_res + 2, self.order, self.v_res, self.order),
                                  (self.x_res, self.u_res, self.order, self.v_res + 2, self.order)]
        self.sub_elements = [2, 4]
        self.directions = [1, 3]

        # arrays
        self.field_flux = var.Distribution(resolutions=resolutions, order=order)
        self.output = var.Distribution(resolutions=resolutions, order=order)

        # magnetic field
        self.b_field = 1.0 / om_pc  # a constant

        self.pad_field = None
        self.pad_spectrum = None

    def initialize_zero_pad(self, grid):
        self.pad_field = cp.zeros(grid.x.modes + grid.x.pad_width) + 0j
        self.pad_spectrum = cp.zeros((grid.x.modes + grid.x.pad_width,
                                     grid.u.elements, grid.u.order,
                                     grid.v.elements, grid.v.order)) + 0j

        # self.pad_field = cp.pad(elliptic.field.arr_spectral, grid.x.pad_width)
        # self.pad_spectrum = cp.pad(distribution.arr,
        #                            grid.x.pad_width)[:, grid.x.pad_width:-grid.x.pad_width,
        #                     grid.x.pad_width:-grid.x.pad_width,
        #                     grid.x.pad_width:-grid.x.pad_width,
        #                     grid.x.pad_width:-grid.x.pad_width]

    def semi_discrete_rhs(self, distribution, elliptic, grid):
        """ Computes the semi-discrete equation """
        # Do elliptic problem
        # t0 = timer.time()
        elliptic.poisson_solve(distribution=distribution, grid=grid, invert=False)
        # t1 = timer.time()
        # # Compute the flux
        self.compute_flux(distribution=distribution, elliptic=elliptic, grid=grid)
        # t2 = timer.time()
        self.output.arr = (grid.u.J * self.u_flux_lgl(distribution=distribution, grid=grid) +
                           grid.v.J * self.v_flux_lgl(distribution=distribution, grid=grid) +
                           self.source_term_lgl(distribution=distribution, grid=grid))
        # a = grid.u.J * self.u_flux_lgl(distribution=distribution, grid=grid)
        # t3 = timer.time()
        # b = grid.v.J * self.v_flux_lgl(distribution=distribution, grid=grid)
        # t4 = timer.time()
        # c = self.source_term_lgl(distribution=distribution, grid=grid)
        # t5 = timer.time()
        # print('Poisson time {:0.3e}'.format(t1 - t0))
        # print('Flux time {:0.3e}'.format(t2 - t1))
        # print('u-flux time {:0.3e}'.format(t3 - t2))
        # print('v-flux time {:0.3e}'.format(t4 - t3))
        # print('source term time {:0.3e}'.format(t5 - t4))
        # quit()
        # return self.output.arr
        # if not gl:
        #     self.output.arr = (grid.v.J * self.v_flux_lgl(grid=grid))
        # self.output.arr = self.source_term(distribution=distribution, grid=grid)

    def compute_flux(self, distribution, elliptic, grid):
        """ Compute the flux convolution(field, distribution) using pseudospectral method """
        # Zero-pad spectrum
        # t0 = timer.time()
        # self.pad_field[grid.x.pad_width:-grid.x.pad_width] = elliptic.field.arr_spectral
        # t1 = timer.time()
        # self.pad_spectrum[grid.x.pad_width:-grid.x.pad_width] = distribution.arr
        # self.pad_field = elliptic.field.arr_spectral
        # self.pad_spectrum = distribution.arr
        # # Pseudospectral product
        # field_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.pad_field, axes=0), norm='forward', axis=0))
        # # t2 = timer.time()
        # distr_nodal = cp.real(cp.fft.ifft(cp.fft.fftshift(self.pad_spectrum, axes=0), norm='forward', axis=0))
        # # t3 = timer.time()
        # nodal_flux = cp.multiply(field_nodal[:, None, None, None, None], distr_nodal)
        # # t4 = timer.time()
        # self.field_flux.arr = cp.fft.fftshift(cp.fft.fft(
        #     nodal_flux, axis=0, norm='forward'), axes=0
        # )  # [grid.x.pad_width:-grid.x.pad_width, :, :, :, :]
        # t5 = timer.time()
        # print('Pad time {:0.3e}'.format(t1 - t0))
        # print('Field transform time {:0.3e}'.format(t2 - t1))
        # print('Distribution transform time {:0.3e}'.format(t3 - t2))
        # print('Multiply time {:0.3e}'.format(t4 - t3))
        # print('Flux fft time {:0.3e}'.format(t5 - t4))
        # quit()

        # self.pad_spectrum[grid.x.pad_width:-grid.x.pad_width] = distribution.arr
        self.pad_field[:-grid.x.pad_width] = elliptic.field.arr_spectral
        self.pad_spectrum[:-grid.x.pad_width, :, :, :, :] = distribution.arr
        # Pseudospectral product
        field_nodal = cp.fft.irfft(self.pad_field, norm='forward', axis=0)
        # t2 = timer.time()
        distr_nodal = cp.fft.irfft(self.pad_spectrum, norm='forward', axis=0)
        # t3 = timer.time()
        nodal_flux = cp.multiply(field_nodal[:, None, None, None, None], distr_nodal)
        # t4 = timer.time()
        self.field_flux.arr = cp.fft.rfft(nodal_flux, axis=0, norm='forward')[:-grid.x.pad_width, :, :, :, :]

    def u_flux_lgl(self, distribution, grid):
        u_flux = (-1.0 * self.field_flux.arr -
                  self.b_field * cp.multiply(grid.v.device_arr[None, None, None, :, :], distribution.arr))
        # plotter = my_plt.Plotter(grid=grid)
        # plotter.velocity_contourf(dist_slice=cp.real(u_flux[4, :, :, :, :]))
        # plotter.show()
        return (basis_product(flux=u_flux, basis_arr=grid.u.local_basis.internal,
                              axis=2, permutation=self.permutations[0]) -
                self.numerical_flux_lgl(flux=u_flux, grid=grid, dim=0))

    def v_flux_lgl(self, distribution, grid):
        v_flux = self.b_field * grid.u.device_arr[None, :, :, None, None] * distribution.arr
        # plotter = my_plt.Plotter(grid=grid)
        # plotter.velocity_contourf(dist_slice=cp.real(v_flux[4, :, :, :, :]))
        # plotter.show()
        return (basis_product(flux=v_flux, basis_arr=grid.v.local_basis.internal,
                              axis=4, permutation=self.permutations[1]) -
                self.numerical_flux_lgl(flux=v_flux, grid=grid, dim=1))

    def numerical_flux_lgl(self, flux, grid, dim):
        # Allocate
        num_flux = cp.zeros(self.num_flux_sizes[dim]) + 0j

        # set padded flux
        padded_flux = cp.zeros(self.padded_flux_sizes[dim]) + 0j
        padded_flux[self.flux_input_slices[dim]] = flux
        # padded_flux[:, 0, -1, :, :] = flux[:, 0, 0, :, :]  # -self.field_flux.arr[:, 0, 0]
        # padded_flux[:, -1, 0, :, :] = flux[:, -1, 0, :, :]  # -self.field_flux.arr[:, -1, 0]
        # padded_flux[:, 0, -1, :, :] = flux[:, 0, 0, :, :]

        # Compute a central flux
        num_flux[self.boundary_slices[dim][0]] = -1.0 * (cp.roll(padded_flux[self.boundary_slices_pad[dim][1]],
                                                                 shift=+1,
                                                                 axis=self.directions[dim])[self.pad_slices[dim]] +
                                                         flux[self.boundary_slices[dim][0]]) / 2.0
        num_flux[self.boundary_slices[dim][1]] = (cp.roll(padded_flux[self.boundary_slices_pad[dim][0]],
                                                          shift=-1,
                                                          axis=self.directions[dim])[self.pad_slices[dim]] +
                                                  flux[self.boundary_slices[dim][1]]) / 2.0

        return basis_product(flux=num_flux, basis_arr=grid.v.local_basis.numerical,
                             axis=self.sub_elements[dim], permutation=self.permutations[dim])

    def source_term_lgl(self, distribution, grid):
        return -1j * cp.multiply(grid.x.device_wavenumbers[:, None, None, None, None],
                                 cp.einsum('ijk,mikrs->mijrs', grid.u.translation_matrix, distribution.arr))
        # cp.tensordot(grid.u.translation_matrix, distribution.arr, axes=([0, 1, 2], []))
