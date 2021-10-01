import numpy as np
# import cupy as np
import grid as g
import variables as var
import elliptic as ell
import plotter as my_plt
import fluxes as fx
import time as timer
import timestep as ts
from copy import deepcopy

# elements and order
elements, order = [8, 25, 25], 6

# set up grid
lows = np.array([-2.0 * np.pi, -6, -6])
highs = np.array([2.0 * np.pi, 6, 6])
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
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.show()

plotter3d = my_plt.Plotter3D(grid=grid)
plotter3d.distribution_contours3d(distribution=distribution, contours=[0.01, 0.1])
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025, 0.05, 0.1],
#                               option='imag')

# Set up fluxes
flux = fx.DGFlux(resolutions=elements, order=order)

# Set up timestepper
dt = 5.0e-3
step = 5.0e-3
final_time = 2.0
steps = int(final_time // step)
dt_max = 1.0 / (np.amax(grid.x.wavenumbers) * np.amax(grid.v.arr))
stepper = ts.Stepper(dt=dt, step=step, resolutions=elements, order=order, steps=steps)
stepper.main_loop(distribution=distribution, elliptic=elliptic,
                  grid=grid, plotter=plotter, plot=False)

plotter.spatial_scalar_plot(scalar=distribution.zero_moment, y_axis='Zero moment')
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy,
                         y_axis='Electric energy', log=True, give_rate=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy + stepper.thermal_energy,
                         y_axis='Total energy', log=False)
plotter.show()

plotter3d.distribution_contours3d(distribution=distribution, contours=[0.01, 0.1])
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025, 0.05, 0.1],
#                               option='imag')
