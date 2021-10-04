import numpy as np
# import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.integrate as integrate
# import pyvista as pv
# import cupy as cp
import scipy.optimize as opt
from numpy.polynomial import Laguerre as lag_poly


quad_arr = np.array([[1, -0.9988664044200710501855,  0.002908622553155140958],
    [2,   -0.994031969432090712585,    0.0067597991957454015028],
    [3,   -0.985354084048005882309,   0.0105905483836509692636],
    [4,   -0.9728643851066920737133,   0.0143808227614855744194],
    [5,   -0.9566109552428079429978,   0.0181155607134893903513],
    [6,   -0.9366566189448779337809,   0.0217802431701247929816],
    [7,   -0.9130785566557918930897,   0.02536067357001239044],
    [8,   -0.8859679795236130486375,   0.0288429935805351980299],
    [9,   -0.8554297694299460846114,   0.0322137282235780166482],
    [10,  -0.821582070859335948356,    0.0354598356151461541607],
    [11,  -0.784555832900399263905,    0.0385687566125876752448],
    [12,  -0.744494302226068538261,    0.041528463090147697422],
    [13,  -0.70155246870682225109,    0.044327504338803275492],
    [14,  -0.6558964656854393607816,   0.0469550513039484329656],
    [15,  -0.6077029271849502391804,   0.0494009384494663149212],
    [16,  -0.5571583045146500543155,   0.0516557030695811384899],
    [17,  -0.5044581449074642016515,   0.0537106218889962465235],
    [18,  -0.449806334974038789147,   0.05555774480621251762357],
    [19,  -0.3934143118975651273942,   0.057189925647728383723],
    [20,  -0.335500245419437356837,    0.058600849813222445835],
    [21,  -0.2762881937795319903276,   0.05978505870426545751],
    [22,  -0.2160072368760417568473,   0.0607379708417702160318],
    [23,  -0.1548905899981459020716,   0.06145589959031666375641],
    [24,  -0.0931747015600861408545,   0.0619360674206832433841],
    [25,  -0.0310983383271888761123,   0.062176616655347262321],
    [26,  0.0310983383271888761123,    0.062176616655347262321],
    [27,  0.09317470156008614085445,   0.0619360674206832433841],
    [28,  0.154890589998145902072,    0.0614558995903166637564],
    [29,  0.2160072368760417568473,    0.0607379708417702160318],
    [30,  0.2762881937795319903276,    0.05978505870426545751],
    [31,  0.335500245419437356837,    0.058600849813222445835],
    [32,  0.3934143118975651273942,    0.057189925647728383723],
    [33,  0.4498063349740387891471,    0.055557744806212517624],
    [34,  0.5044581449074642016515,    0.0537106218889962465235],
    [35,  0.5571583045146500543155,    0.05165570306958113849],
    [36,  0.60770292718495023918,     0.049400938449466314921],
    [37,  0.6558964656854393607816,    0.046955051303948432966],
    [38,  0.7015524687068222510896,    0.044327504338803275492],
    [39,  0.7444943022260685382605,    0.0415284630901476974224],
    [40,  0.7845558329003992639053,    0.0385687566125876752448],
    [41,  0.8215820708593359483563,    0.0354598356151461541607],
    [42,  0.8554297694299460846114,    0.0322137282235780166482],
    [43,  0.8859679795236130486375,    0.02884299358053519803],
    [44,  0.9130785566557918930897,    0.02536067357001239044],
    [45,  0.9366566189448779337809,    0.0217802431701247929816],
    [46,  0.9566109552428079429978,    0.0181155607134893903513],
    [47,  0.9728643851066920737133,    0.0143808227614855744194],
    [48,  0.985354084048005882309,    0.010590548383650969264],
    [49,  0.9940319694320907125851,    0.0067597991957454015028],
    [50,  0.9988664044200710501855,    0.0029086225531551409584]])


# Parameters
a = 10.0  # 10.0  # 10.0  # 20.0 # omega_p / omega_c
j = 6
# Grids
fr = np.linspace(-1.5, 1.5, num=75)
fi = np.linspace(-0.1, 1.5, num=75)
fz = np.tensordot(fr, np.ones_like(fi), axes=0) + 1.0j * np.tensordot(np.ones_like(fr), fi, axes=0)


def integrand(x, om, wave):
    t = 0.5 * np.pi * (1.0 + x)  # affine transform
    beta = 2.0 * np.power(wave * np.cos(0.5 * t), 2)
    return np.sin(t * om) * np.sin(t) * np.exp(-beta) * lag_poly((*np.zeros(j), 1.0))(beta)


def jac_integrand(x, om, wave):
    t = 0.5 * np.pi * (1.0 + x)  # affine transform
    beta = 2.0 * np.power(wave * np.cos(0.5 * t), 2)
    return t * np.cos(t * om) * np.sin(t) * np.exp(-beta) * lag_poly((*np.zeros(j), 1.0))(beta)
    # deriv = t * np.cos(t * om) - np.pi * np.cos(np.pi * om) * np.sin(om * t) / np.sin(np.pi * om)
    # return deriv * np.sin(t) * np.exp(-beta) * lag_poly((*np.zeros(j), 1.0))(beta)

# def complex_integrate(om, wave):
#     def real_integral(x):
#         return np.real(integrand(x, om, wave))
#
#     def imag_integral(x):
#         return np.imag(integrand(x, om, wave))
#
#     real_int = integrate.quad(real_integral, 0, np.pi)
#     imag_int = integrate.quad(imag_integral, 0, np.pi)
#     return real_int[0] + 1j * imag_int[0]


def dispersion(om, wave):
    inner = integrand(x=quad_arr[:, 1], om=om, wave=wave)
    quad = 0.5 * np.pi * np.tensordot(quad_arr[:, 2], inner, axes=([0], [0]))
    return np.sin(np.pi * om) + (a ** 2.0) * quad
    # return 1.0 + (a ** 2.0) * quad / np.sin(np.pi * om)


def jacobian(om, wave):
    inner = jac_integrand(x=quad_arr[:, 1], om=om, wave=wave)
    quad = 0.5 * np.pi * np.tensordot(quad_arr[:, 2], inner, axes=([0], [0]))
    return np.pi * np.cos(np.pi * om) + (a ** 2.0) * quad
    # return (a ** 2.0) * quad / np.sin(np.pi * om)


def dispersion_fsolve(om, wave):
    freq = om[0] + 1j*om[1]
    d = dispersion(freq, wave)
    return [np.real(d), np.imag(d)]


def jacobian_fsolve(om, wave):
    freq = om[0] + 1j*om[1]
    jac = jacobian(freq, wave)
    jr, ji = np.real(jac), np.imag(jac)
    return [[jr, -ji], [ji, jr]]


# X, Y = np.meshgrid(fr, fi, indexing='ij')
# arr = np.array([[dispersion(om=fz[i, j], wave=0.888) for j in range(fz.shape[1])] for i in range(fz.shape[0])])
#
# plt.figure()
# plt.contour(X, Y, np.real(arr), 0, colors='g')
# plt.contour(X, Y, np.imag(arr), 0, colors='r')
# plt.show()


waves = np.linspace(0.5, 2.0*np.pi, num=1000)
first_harmonic = np.zeros_like(waves) + 0j
second_harmonic = np.zeros_like(waves) + 0j
third_harmonic = np.zeros_like(waves) + 0j
fourth_harmonic = np.zeros_like(waves) + 0j

guess_r, guess_r2, guess_r3 = np.zeros_like(waves), np.zeros_like(waves), np.zeros_like(waves)
guess_r4 = np.zeros_like(waves)
guess_i = np.zeros_like(waves)

guess_r[waves <= 0.77] = 1.0
guess_r[waves >= 0.77] = 0.0
guess_r[waves >= 1.0] = 1.0
guess_r[waves >= 1.22] = 1.1

guess_i[waves <= 0.77] = 0.2
guess_i[waves >= 0.77] = 0.55
guess_i[waves >= 0.85] = 0.55
guess_i[waves >= 1.1] = 0.3

guess_r2[waves <= 0.6] = 2.5
guess_r2[waves >= 0.6] = 2.2
guess_r2[waves >= 0.75] = 2.1
guess_r2[waves >= 1.0] = 2.0

guess_r3[waves <= 0.75] = 4.0
guess_r3[waves >= 0.75] = 3.5
guess_r3[waves >= 1.2] = 3.0
guess_r3[waves >= 1.5] = 2.5
guess_r3[waves >= 1.75] = 3.0

guess_r4[waves <= 0.75] = 5.0
guess_r4[waves >= 0.75] = 4.5
guess_r4[waves >= 1.2] = 4.0
guess_r4[waves >= 1.5] = 3.5
guess_r4[waves >= 1.75] = 4.0

# first_harmonic[0] = 1.414
# second_harmonic[0] = 2.0
# third_harmonic[0] = 3.0

for idx, k in enumerate(waves):
    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r[idx], guess_i[idx]]), args=k, jac=jacobian_fsolve)
    first_harmonic[idx] = solution.x[0] + 1j * solution.x[1]

    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r2[idx], 0.2]), args=k, jac=jacobian_fsolve)
    second_harmonic[idx] = solution.x[0] + 1j * solution.x[1]

    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r3[idx], 0]), args=k, jac=jacobian_fsolve)
    third_harmonic[idx] = solution.x[0] + 1j * solution.x[1]

    solution = opt.root(dispersion_fsolve, x0=np.array([guess_r4[idx], 0]), args=k, jac=jacobian_fsolve)
    fourth_harmonic[idx] = solution.x[0] + 1j * solution.x[1]

plt.figure()
L = np.zeros_like(waves)
L[1:] = 2.0 * np.pi / waves[1:]
L[0] = L[1]
plt.plot(L, np.real(first_harmonic), 'k')
plt.plot(L, np.imag(first_harmonic), 'r')
plt.plot(L, np.real(second_harmonic), 'k')
plt.plot(L, np.real(third_harmonic), 'k')
plt.plot(L, np.real(fourth_harmonic), 'k')
plt.grid(True)
# plt.xlabel(r'Wavenumber $kr_L$'), plt.ylabel(r'Frequency $\omega / \omega_c$')
plt.xlabel(r'Wavelength $L/\lambda_L$'), plt.ylabel(r'Frequency $\omega / \omega_c$')
# plt.axis([waves[0], waves[-1], 0, 4.5])
plt.axis([L[-1], L[0], -0.1, 5.1])
plt.show()

# A particular solution
wave = 0.888
print(opt.fsolve(dispersion_fsolve, x0=[0.0, 0.348], args=wave))
# print(opt.fsolve(dispersion_fsolve, x0=[2.000001, 0], args=wave))
# print(opt.fsolve(dispersion_fsolve, x0=[3.000001, 0], args=wave))

quit()

X, Y = np.meshgrid(fr, fi, indexing='ij')
arr = np.array([[dispersion(om=fz[i, j], wave=2.0) for j in range(fz.shape[1])] for i in range(fz.shape[0])])

plt.figure()
plt.contour(X, Y, np.real(arr), 0, colors='g')
plt.contour(X, Y, np.imag(arr), 0, colors='r')
plt.show()

quit()

