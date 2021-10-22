import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

# import scipy.integrate as integrate
# import scipy.optimize as opt
# from numpy.polynomial import Laguerre as lag_poly


ring_j = 0
om_pc = 1  # electron omega_p / omega_c


def master_series(n, b, g, terms):
    if g == -1:
        return 0
    out = 0
    for i in range(terms):
        out += (sp.gamma(2 * (n + i + g) + 1) * (sp.rgamma(n + i + 1) ** 2) * sp.rgamma(2 * n + i + 1) *
                np.power(-1, i) * sp.rgamma(i + 1) * np.power(b, 2 * (i + n)))
    return sp.rgamma(g + 1) * out


def v_series(a, b, g, terms):
    return sum([
        (n / (n - a)) * (master_series(n=np.abs(n), b=b, g=g-1, terms=15) -
                         master_series(n=np.abs(n), b=b, g=g, terms=15))
        for n in range(1 - terms, terms)
    ])


def dispersion(om, k, g, mass_ratio):
    # electron term
    e_term = v_series(a=-1.0 * om, b=k, g=g, terms=10)
    # p_term = v_series(a=mass_ratio*om, b=mass_ratio*k, terms=10)
    return 1.0 - e_term / np.power(k, 2) / 2


# Main section
mass_ratio = 100
k = 0.01
fr = np.linspace(-0.5, 2.5, num=50)
fi = np.linspace(-0.001, 0.3, num=50)
fz = np.tensordot(fr, np.ones_like(fi), axes=0) + 1.0j * np.tensordot(np.ones_like(fr), fi, axes=0)
X, Y = np.meshgrid(fr, fi, indexing='ij')
arr = np.array([[dispersion(om=fz[i, j], k=k, g=0, mass_ratio=mass_ratio)
                 for j in range(fz.shape[1])] for i in range(fz.shape[0])])

plt.figure()
plt.contour(X, Y, np.real(arr), 0, colors='g')
plt.contour(X, Y, np.imag(arr), 0, colors='r')
plt.grid(True), plt.show()
