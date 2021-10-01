import cupy as cp
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, grid, colormap='RdPu'):
        self.colormap = colormap
        self.grid = grid
        # Build structured grid, nodal
        # self.X, self.V = np.meshgrid(grid.x.arr.flatten(), grid.v.arr.flatten(), indexing='ij')
        self.x = grid.x.arr
        self.k = grid.x.wavenumbers / grid.x.fundamental
        # Build structured grid, global spectral
        # self.FX, self.FV = np.meshgrid(grid.x.wavenumbers / grid.x.fundamental, grid.v.device_arr.flatten(),
        #                              indexing='ij')
        # plt.ion()

    def spatial_scalar_plot(self, scalar, y_axis, spectrum=True):
        if scalar.arr_nodal is None:
            scalar.inverse_fourier_transform()

        plt.figure()
        plt.plot(self.x.flatten(), scalar.arr_nodal.flatten().get(), 'o')
        plt.xlabel('x'), plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()

        if spectrum:
            plt.figure()
            spectrum = scalar.arr_spectral.flatten().get()
            plt.plot(self.k.flatten(), np.real(spectrum), 'ro', label='real')
            plt.plot(self.k.flatten(), np.imag(spectrum), 'go', label='imaginary')
            plt.xlabel('Modes'), plt.ylabel(y_axis + ' spectrum')
            plt.grid(True), plt.legend(loc='best'), plt.tight_layout()

    def time_series_plot(self, time_in, series_in, y_axis, log=False, give_rate=False):
        time, series = time_in, series_in.get()
        plt.figure()
        if log:
            plt.semilogy(time, series, 'o--')
        else:
            plt.plot(time, series, 'o--')
        plt.xlabel('Time')
        plt.ylabel(y_axis)
        plt.grid(True), plt.tight_layout()
        if give_rate:
            lin_fit = np.polyfit(time, np.log(series), 1)
            print('Numerical rate: {:0.10e}'.format(lin_fit[0]))
            print('cf. exact rate: {:0.10e}'.format(2 * 2.409497728e-01))
            print('The difference is {:0.10e}'.format(lin_fit[0] - 2 * 2.409497728e-01))

    def show(self):
        plt.show()


class Plotter3D:
    """
    Plots objects on 3D piecewise (as in DG) grid
    """

    def __init__(self, grid):
        # Build structured grid, full space
        (ix, iu, iv) = (cp.ones(grid.x.elements),
                        cp.ones(grid.u.elements * grid.u.order),
                        cp.ones(grid.v.elements * grid.v.order))
        (x3, u3, v3) = (outer3(a=grid.x.device_arr, b=iu, c=iv),
                        outer3(a=ix, b=grid.u.device_arr.flatten(), c=iv),
                        outer3(a=ix, b=iu, c=grid.v.device_arr.flatten()))
        self.grid = pv.StructuredGrid(x3, u3, v3)

        # build structured grid, spectral space
        k3 = outer3(a=grid.x.device_wavenumbers, b=iu, c=iv)
        self.spectral_grid = pv.StructuredGrid(k3, u3, v3)

    def distribution_contours3d(self, distribution, contours):
        """
        plot contours of a scalar function f=f(x,y,z) on Plotter3D's grid
        """
        self.grid['.'] = distribution.grid_flatten().get().transpose().flatten()
        plot_contours = self.grid.contour(contours)

        # Create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True)
        p.show_grid()
        p.show()  # auto_close=False)

    def spectral_contours3d(self, distribution, contours, option='real'):
        """
        plot contours of a scalar function f=f(k,y,z)self.spectral_grid['.'] = np.real(distribution.spectral_flatten().get().transpose().flatten())
         on Plotter3D's spectral grid
        """
        if option == 'real':
            self.spectral_grid['.'] = np.real(distribution.spectral_flatten().get().transpose().flatten())
        if option == 'imag':
            self.spectral_grid['.'] = np.imag(distribution.spectral_flatten().get().transpose().flatten())
        if option == 'absolute':
            self.spectral_grid['.'] = np.absolute(distribution.spectral_flatten().get().transpose().flatten())
        plot_contours = self.spectral_grid.contour(contours)

        # create plot
        p = pv.Plotter()
        p.add_mesh(plot_contours, cmap='summer', show_scalar_bar=True, opacity=0.75)
        p.show_grid()
        p.show()


def outer3(a, b, c):
    """
    Compute outer tensor product of vectors a, b, and c
    :param a: vector a_i
    :param b: vector b_j
    :param c: vector c_k
    :return: tensor a_i b_j c_k as numpy array
    """
    return cp.tensordot(a, cp.tensordot(b, c, axes=0), axes=0).get()
