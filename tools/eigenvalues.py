import numpy as np
# import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
# import pyvista as pv
# import cupy as cp
import scipy.optimize as opt

from numpy.polynomial import Laguerre as lag_poly

# Parameters
a = 1.0  # 10.0  # 10.0  # 20.0 # omega_p / omega_c
j = 0  # 6
# Grids
k = 0.5
fr = np.linspace(0.75, 1.5, num=75)
fi = np.linspace(-0.1, 0.1, num=75)
fz = np.tensordot(fr, np.ones_like(fi), axes=0) + 1.0j * np.tensordot(np.ones_like(fr), fi, axes=0)


def integrand(x, om, wave):
    beta = 2.0 * np.power(wave * np.cos(0.5 * x), 2)
    return np.sin(x * om) * np.sin(x) * np.exp(-beta)  # * lag_poly((*np.zeros(j), 1.0))(beta)


def complex_integrate(om, wave):
    def real_integral(x):
        return np.real(integrand(x, om, wave))

    def imag_integral(x):
        return np.imag(integrand(x, om, wave))

    real_int = integrate.quad(real_integral, 0, np.pi)
    imag_int = integrate.quad(imag_integral, 0, np.pi)
    return real_int[0] + 1j * imag_int[0]


def dispersion(om, wave):
    return 1.0 + (a ** 2.0) / np.sin(np.pi * om) * complex_integrate(om, wave)


def dispersion_fsolve(om, wave):
    freq = om[0] + 1j*om[1]
    d = dispersion(freq, wave)
    return [np.real(d), np.imag(d)]


solution = opt.fsolve(dispersion_fsolve, x0=[1.16, 0], args=k)
print(solution)

X, Y = np.meshgrid(fr, fi, indexing='ij')
arr = np.array([[dispersion(om=fz[i, j], wave=1.0) for j in range(fz.shape[1])] for i in range(fz.shape[0])])


plt.figure()
plt.contour(X, Y, np.real(arr), 0, colors='g')
plt.contour(X, Y, np.imag(arr), 0, colors='r')
plt.show()
