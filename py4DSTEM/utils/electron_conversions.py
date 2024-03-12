import math as ma


def electron_wavelength_angstrom(E_eV):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34

    lam = (
        h
        / ma.sqrt(2 * m * e * E_eV)
        / ma.sqrt(1 + e * E_eV / 2 / m / c**2)
        * 10**10
    )
    return lam


def electron_interaction_parameter(E_eV):
    m = 9.109383 * 10**-31
    e = 1.602177 * 10**-19
    c = 299792458
    h = 6.62607 * 10**-34
    lam = (
        h
        / ma.sqrt(2 * m * e * E_eV)
        / ma.sqrt(1 + e * E_eV / 2 / m / c**2)
        * 10**10
    )
    sigma = (
        (2 * np.pi / lam / E_eV) * (m * c**2 + e * E_eV) / (2 * m * c**2 + e * E_eV)
    )
    return sigma


