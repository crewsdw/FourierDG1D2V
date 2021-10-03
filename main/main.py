import numpy as np
import cupy as cp
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import fluxes as fx
import time as timer
import timestep as ts
from copy import deepcopy

# elements and order
elements, order = [10, 30, 30], 8

# set up grid
lows = np.array([-0.5 * np.pi, -5, -5])
highs = np.array([0.5 * np.pi, 5, 5])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order)

# build distribution
distribution = var.Distribution(resolutions=elements, order=order)
distribution.initialize(grid=grid)
distribution.fourier_transform()
distribution.inverse_fourier_transform()

# test elliptic solver
elliptic = ell.Elliptic(resolution=elements[0])
elliptic.poisson_solve(distribution=distribution, grid=grid)

# test plotters
plotter = my_plt.Plotter(grid=grid)
plotter.spatial_scalar_plot(scalar=distribution.zero_moment, y_axis='Zero moment')
# plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.velocity_contourf(dist_slice=cp.imag(distribution.arr[6, :, :, :, :]))
plotter.velocity_contourf(dist_slice=cp.real(distribution.arr[6, :, :, :, :]))
plotter.show()

plotter3d = my_plt.Plotter3D(grid=grid)
# plotter3d.distribution_contours3d(distribution=distribution, contours=[0.01, 0.1])
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025, 0.05, 0.1],
#                               option='imag')

# Set up fluxes
flux = fx.DGFlux(resolutions=elements, order=order)

# Set up time-stepper
dt = 1.0e-3
step = 1.0e-3
final_time = 6.0
steps = int(final_time // step)
dt_max = 1.0 / (np.amax(grid.x.wavenumbers) * np.amax(grid.v.arr))
stepper = ts.Stepper(dt=dt, step=step, resolutions=elements, order=order, steps=steps)
stepper.main_loop(distribution=distribution, elliptic=elliptic,
                  grid=grid, plotter=plotter, plot=False)

plotter.spatial_scalar_plot(scalar=distribution.zero_moment, y_axis='Zero moment')
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy,
                         y_axis='Electric energy', log=False, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy + stepper.thermal_energy,
                         y_axis='Total energy', log=False)
# plotter.velocity_contourf(dist_slice=distribution.arr_nodal[3, :, :, :, :])
plotter.velocity_contourf(dist_slice=cp.imag(distribution.arr[6, :, :, :, :]))
plotter.velocity_contourf(dist_slice=cp.real(distribution.arr[6, :, :, :, :]))
plotter.velocity_contourf(dist_slice=cp.imag(distribution.arr[7, :, :, :, :]))
plotter.velocity_contourf(dist_slice=cp.real(distribution.arr[7, :, :, :, :]))
plotter.velocity_contourf(dist_slice=cp.imag(distribution.arr[8, :, :, :, :]))
plotter.velocity_contourf(dist_slice=cp.real(distribution.arr[8, :, :, :, :]))
plotter.show()

distribution.arr[4, :, :, :, :] = 0
distribution.arr[5, :, :, :, :] = 0
distribution.arr[6, :, :, :, :] = 0
distribution.inverse_fourier_transform()

contours = np.linspace(cp.amin(distribution.arr_nodal).get(), cp.amax(distribution.arr_nodal).get(), num=10)
print(contours)

plotter3d.distribution_contours3d(distribution=distribution, contours=contours)
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025q, 0.05, 0.1],
#                               option='imag')
