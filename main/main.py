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
elements, order = [32, 40, 40], 8

# set up grid
om_pc = 10.0
wave_number = 0.888 / om_pc
length = 2.0 * np.pi / wave_number
lows = np.array([-0.5 * length, -8.5, -8.5])
highs = np.array([0.5 * length, 8.5, 8.5])
grid = g.PhaseSpace(lows=lows, highs=highs, elements=elements, order=order, om_pc=om_pc)

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
plotter.velocity_contourf_complex(dist_slice=distribution.arr[0, :, :, :, :], title='Mode 0')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[1, :, :, :, :], title='Mode 1')
plotter.show()

plotter3d = my_plt.Plotter3D(grid=grid)
# plotter3d.distribution_contours3d(distribution=distribution, contours=[0.01, 0.1])
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025, 0.05, 0.1],
#                               option='imag')

# Set up fluxes
flux = fx.DGFlux(resolutions=elements, order=order, om_pc=om_pc)
flux.initialize_zero_pad(grid=grid)

# Set up time-stepper
print('Lorentz force dt estimate:{:0.3e}'.format(1.0/(np.sqrt(2)*highs[1]/om_pc)))
print('Spatial flux dt estimate:{:0.3e}'.format(1.0/(np.sqrt(2)*highs[1]*grid.x.wavenumbers[-1])))
dt = 8.409e-03  # 1.025e-02 * 1.0
step = 8.409e-03  # 1.025e-02 * 1.0
final_time = 110.0
steps = int(final_time // step) + 1
dt_max = 1.0 / (np.amax(grid.x.wavenumbers) * np.amax(grid.v.arr))
stepper = ts.Stepper(dt=dt, step=step, resolutions=elements, order=order, steps=steps, flux=flux)
stepper.main_loop(distribution=distribution, elliptic=elliptic,
                  grid=grid, plotter=plotter, plot=False)

distribution.zero_moment.inverse_fourier_transform()
plotter.spatial_scalar_plot(scalar=distribution.zero_moment, y_axis='Zero moment')
plotter.spatial_scalar_plot(scalar=elliptic.field, y_axis='Electric Field')
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy,
                         y_axis='Electric energy', log=True, give_rate=True)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.thermal_energy,
                         y_axis='Thermal energy', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.density_array,
                         y_axis='Total density', log=False)
plotter.time_series_plot(time_in=stepper.time_array, series_in=stepper.field_energy + stepper.thermal_energy,
                         y_axis='Total energy', log=False)

plotter.velocity_contourf_complex(dist_slice=distribution.arr[0, :, :, :, :], title='Mode 0')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[1, :, :, :, :], title='Mode 1')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[2, :, :, :, :], title='Mode 2')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[3, :, :, :, :], title='Mode 3')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[4, :, :, :, :], title='Mode 4')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[5, :, :, :, :], title='Mode 5')
plotter.velocity_contourf_complex(dist_slice=distribution.arr[6, :, :, :, :], title='Mode 6')

plotter.show()

# distribution.arr[7, :, :, :, :] = 0
# distribution.arr[8, :, :, :, :] = 0
# distribution.arr[9, :, :, :, :] = 0
distribution.inverse_fourier_transform()

contours = np.linspace(1.2 * np.absolute(cp.amin(distribution.arr_nodal).get()),
                       0.3 * cp.amax(distribution.arr_nodal).get(), num=10)
# print(contours)

plotter3d.distribution_contours3d(distribution=distribution, contours=contours)
# plotter3d.spectral_contours3d(distribution=distribution, contours=[-0.025, -0.01, 0.01, 0.025q, 0.05, 0.1],
#                               option='imag')
