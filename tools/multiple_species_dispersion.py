import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.optimize as opt

ring_j = 0
om_pc = 1  # electron omega_p / omega_c


def master_series(n, b, g, terms):
    if g == -1:
        return 0
    out = 0
    for i in range(terms):
        out += (sp.gamma(2 * (n + i + g) + 1) * (sp.rgamma(n + i + 1) ** 2) * sp.rgamma(2 * n + i + 1) *
                np.power(-1, i) * sp.rgamma(i + 1) * np.power(b, 2 * (i + n)))
        term = (sp.gamma(2 * (n + i + g) + 1) * (sp.rgamma(n + i + 1) ** 2) * sp.rgamma(2 * n + i + 1) *
                np.power(-1, i) * sp.rgamma(i + 1) * np.power(b, 2 * (i + n)))
        # print(term)

    return sp.rgamma(g + 1) * out


def v_series(a, e_series, terms):
    vals = range(1 - terms, terms)
    return sum([
        (n / (n - a)) * e_series[idx]
        for idx, n in enumerate(vals)
    ])


def v_series_prime(a, e_series, terms):
    vals = range(1 - terms, terms)
    return sum([
        n / np.power(n - a, 2) * e_series[idx]
        for idx, n in enumerate(vals)
    ])


def dispersion(om, k, g, mass_ratio):
    terms = 20
    e_terms = 60
    # electrons
    b = -k
    a = -om
    # compute the series
    e_series = np.array([
        (master_series(n=np.abs(n), b=b, g=g - 1, terms=e_terms) -
         master_series(n=np.abs(n), b=b, g=g, terms=e_terms)) / 2
        for n in range(1 - terms, terms)
    ])

    e_term = v_series(a=a, e_series=e_series, terms=terms)

    # protons
    b = np.sqrt(mass_ratio) * k
    a = mass_ratio * om
    p_series = np.array([
        (master_series(n=np.abs(n), b=b, g=g - 1, terms=e_terms) -
         master_series(n=np.abs(n), b=b, g=g, terms=e_terms)) / 2
        for n in range(1 - terms, terms)
    ])

    p_term = v_series(a=a, e_series=p_series, terms=terms)

    return 1.0 - (e_term + p_term) / np.power(k, 2)


def jacobian(om, k, g, mass_ratio):
    terms = 10
    e_terms = 50
    # electrons
    b = -k
    a = -om
    # compute the series
    e_series = np.array([
        (master_series(n=np.abs(n), b=b, g=g - 1, terms=e_terms) -
         master_series(n=np.abs(n), b=b, g=g, terms=e_terms)) / 2
        for n in range(1 - terms, terms)
    ])

    e_term = v_series_prime(a=a, e_series=e_series, terms=terms)

    # protons
    b = np.sqrt(mass_ratio) * k
    a = mass_ratio * om
    p_series = np.array([
        (master_series(n=np.abs(n), b=b, g=g - 1, terms=e_terms) -
         master_series(n=np.abs(n), b=b, g=g, terms=e_terms)) / 2
        for n in range(1 - terms, terms)
    ])

    p_term = v_series_prime(a=a, e_series=p_series, terms=terms) / (-1.0 * mass_ratio)

    return 1.0 * (e_term + p_term) / np.power(k, 2)


def dispersion_fsolve(om, k, g, mass_ratio):
    freq = om[0] + 1j * om[1]
    d = dispersion(om=freq, k=k, g=g, mass_ratio=mass_ratio)
    return np.array([np.real(d), np.imag(d)])


def jacobian_fsolve(om, k, g, mass_ratio):
    freq = om[0] + 1j * om[1]
    jac = jacobian(om=freq, k=k, g=g, mass_ratio=mass_ratio)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]  # using cauchy-riemann equations


# Main section
mass_ratio = 1836
k = 0.02
fr = np.linspace(-2e-4, 0.15, num=3000)
fi = np.linspace(-6e-5, 0.2, num=3000)
fz = np.tensordot(fr, np.ones_like(fi), axes=0) + 1.0j * np.tensordot(np.ones_like(fr), fi, axes=0)
X, Y = np.meshgrid(fr, fi, indexing='ij')
arr = dispersion(om=fz, k=k, g=0, mass_ratio=mass_ratio)

plt.figure()
plt.contour(X, Y, np.real(arr), 0, colors='g')
plt.contour(X, Y, np.imag(arr), 0, colors='r')
plt.xlabel(r'$\omega_r/\omega_{ce}$'), plt.ylabel(r'$\omega_i/\omega_{ce}$')
plt.grid(True), plt.tight_layout(), plt.show()

# Root analysis: single root
guess_r, guess_i = 0.0165, 0  # 0.56, -0.001
solution = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                    args=(k, 0, mass_ratio), jac=jacobian_fsolve)

waves = np.linspace(0.01, 0.25, num=20)
freqs = np.zeros_like(waves) + 0j

for idx, wave in enumerate(waves):
    sol = opt.root(dispersion_fsolve, x0=np.array([guess_r, guess_i]),
                   args=(wave, 0, mass_ratio), jac=jacobian_fsolve)
    guess_r, guess_i = sol.x[0], sol.x[1]
    freqs[idx] = sol.x[0] + 1j * sol.x[1]

plt.figure()
plt.plot(waves, np.real(freqs), 'k--')
plt.xlabel(r'Wavenumber $k$'), plt.ylabel(r'Frequency $\omega_r$')
plt.grid(True), plt.tight_layout()

plt.show()
