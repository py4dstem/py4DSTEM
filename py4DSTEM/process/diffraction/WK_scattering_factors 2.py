import numpy as np
from scipy.special import expi

# from functools import lru_cache

from ..utils import electron_wavelength_angstrom

"""
Weickenmeier-Kohl absorptive scattering factors, adapted by SE Zeltmann from EMsoftLib/others.f90 
by Mark De Graef, who adapted it from Weickenmeier's original f77 code.
"""


# @lru_cache(maxsize=1024)
def compute_WK_factor(
    g: float,
    Z: int,
    accelerating_voltage: float,
    thermal_sigma: float = None,
    include_core: bool = True,
    include_phonon: bool = True,
    verbose=False,
) -> np.complex128:
    """
    Compute the Weickenmeier-Kohl atomic scattering factors, using the parameterization
    of the elastic part and computation of the inelastic part found in EMsoftLib/others.f90.
    Return value should be in VÅ.

    This implementation always returns the absorptive, relativistically corrected factors.

    Currently this is mostly a direct translation of the Fortran code, along with
    the accompanying comments from the original in quotation marks. This could be
    vectorized and refactored for efficiency.

    This method uses an 8-parameter fit to the elastic form factors, and then computes the
    absorptive form factors using an analytic solution based on that fitting function. EMsoft would
    use this routine to precompute and cache the necessary factors before calculating U_g.
    I follow this route, but use Python's native caching to wrap this function rather than
    build the lookup table manually. When calling this inside the loop that computes
    the lattice Fourier coefficients, the cache will quickly build up.

    Args: (note that these values cannot be arrays: the code is not vectorized)
        g (float):                      Scattering vector magnitude in the crystallographic/py4DSTEM
                                        convention, 1/d_hkl in units of 1/Å
        Z (int):                        Atomic number. Data are available for H thru Cf (1 thru 98)
        accelerating_voltage (float):   Accelerating voltage in eV.
        thermal_sigma (float):          RMS atomic displacement for TDS, in Å
                                        (This is often written as 〈u〉in papers)
        include_core (bool):            If True, include the core loss contribution to the absorptive
                                        form factors.
        include_phonon (bool):          If True, include the phonon/TDS contribution to the
                                        absorptive form factors.
    Returns:
        Fscatt (np.complex128):         The computed atomic form factor
    """

    # the WK Fortran code works in weird units:
    # lowercase "g", our input, is the standard crystallographic quantity, in Å^-1
    # uppercase "G" is the "G" in others.f90:FSCATT, g * 2π
    # uppercase "S" is the "S" in others.f90:FSCATT, G / 4π = g / 2
    G = g * 2.0 * np.pi
    S = g / 2.0

    if verbose:
        print(f"S:{S}")

    accelerating_voltage_kV = accelerating_voltage / 1.0e3

    if thermal_sigma is not None:
        UL = thermal_sigma
        DWF = np.exp(-0.5 * UL ** 2 * G ** 2)
    else:
        UL = 0.0
        DWF = 1.0

    if verbose:
        print(f"DWF:{DWF}")

    A = WK_A_param[int(Z) - 1]
    B = WK_B_param[int(Z) - 1]

    if verbose:
        print(f"A:{A}")
        print(f"B:{B}")

    # WEKO(A,B,S)
    WK = 0.0
    for i in range(4):
        argu = B[i] * S ** 2
        if argu < 1.0:
            WK += A[i] * B[i] * (1.0 - 0.5 * argu)
        elif argu > 20.0:
            WK += A[i] / S ** 2
        else:
            WK += A[i] * (1.0 - np.exp(-argu)) / S ** 2

    Freal = 4.0 * np.pi * DWF * WK

    if verbose:
        print(f"Freal:{Freal}")

    #################################################
    # calculate "core" contribution, following FCORE:
    k0 = (
        2.0 * np.pi / electron_wavelength_angstrom(accelerating_voltage)
    )  # remember, physicist units here

    if include_core:
        # "CALCULATE CHARACTERISTIC ENERGY LOSS AND ANGLE"
        DE = 6.0e-3 * Z
        theta_e = (
            DE
            / (2.0 * accelerating_voltage_kV)
            * (2.0 * accelerating_voltage_kV + 1022.0)
            / (accelerating_voltage_kV + 1022.0)
        )

        # "SCREENING PARAMETER OF YUKAWA POTENTIAL"
        R = 0.885 * 0.5289 / Z ** (1.0 / 3.0)

        # "CALCULATE NORMALISING ANGLE"
        TA = 1.0 / (k0 * R)

        # "CALCULATE BRAGG ANGLE"
        TB = G / (2.0 * k0)

        # "NORMALIZE"
        OMEGA = 2.0 * TB / TA
        KAPPA = theta_e / TA

        K2 = KAPPA * KAPPA
        O2 = OMEGA * OMEGA

        X1 = (
            OMEGA
            / ((1.0 + O2) * np.sqrt(O2 + 4.0 * K2))
            * np.log((OMEGA + np.sqrt(O2 + 4.0 * K2)) / (2.0 * KAPPA))
        )
        X2 = (
            1.0
            / np.sqrt((1.0 + O2) * (1.0 + O2) + 4.0 * K2 * O2)
            * np.log(
                (1.0 + 2.0 * K2 + O2 + np.sqrt((1.0 + O2) * (1.0 + O2) + 4.0 * K2 * O2))
                / (2.0 * KAPPA * np.sqrt(1.0 + K2))
            )
        )
        if OMEGA > 1e-2:
            X3 = (
                1.0
                / (OMEGA * np.sqrt(O2 + 4.0 * (1.0 + K2)))
                * np.log(
                    (OMEGA + np.sqrt(O2 + 4.0 * (1.0 + K2))) / (2.0 * np.sqrt(1.0 + K2))
                )
            )
        else:
            X3 = 1.0 / (4.0 * (1.0 + K2))

        HI = 2 * Z / (TA * TA) * (-X1 + X2 - X3)

        A0 = 0.5289
        Fcore = 4.0 / (A0 * A0) * 2.0 * np.pi / (k0 * k0) * HI

        if verbose:
            print(f"Fcore:{Fcore}")
    else:
        Fcore = 0.0

    ##########################################################
    # calculate phonon contribution, following FPHON(G,UL,A,B)
    Fphon = 0.0
    if include_phonon:
        U2 = UL ** 2

        A1 = A * (4.0 * np.pi) ** 2
        B1 = B / (4.0 * np.pi) ** 2

        for jj in range(4):
            Fphon += (
                A1[jj]
                * A1[jj]
                * (
                    DWF * RI1(B1[jj], B1[jj], G) - RI2(B1[jj], B1[jj], G, UL)
                )  # should this be + or - ? others.f90 appears to disagree with WK paper...
            )
            for ii in range(jj + 1):
                Fphon += (
                    2.0
                    * A1[jj]
                    * A1[ii]
                    * (
                        DWF * RI1(B1[ii], B1[jj], G) - RI2(B1[ii], B1[jj], G, UL)
                    )  # should this be + or - ? others.f90 appears to disagree with WK paper...
                )
        if verbose:
            print(f"Fphon:{Fphon}")

    Fimag = (Fcore * DWF) + Fphon

    # perform relativistic correction
    gamma = (accelerating_voltage_kV + 511.0) / (511.0)

    if verbose:
        print(f"gamma:{gamma}")

    Fscatt = np.complex128((Freal * gamma) + (1.0j * (Fimag * gamma ** 2 / k0)))

    if verbose:
        print(f"Fscatt:{Fscatt}")

    return (
        Fscatt * 0.4787801 * 0.664840340614319 / (4.0 * np.pi)
    )  # convert to Volts, and remove extra physicist factors, as performed in diffraction.f90:427,576,630
    # I am really not certain about these factors! AARGH


##############################################
# Helper integral functions for DW calculation


def RI1(BI, BJ, G):
    # "ERSTES INTEGRAL FUER DIE ABSORPTIONSPOTENTIALE"
    eps = np.max([BI, BJ]) * G ** 2

    if eps <= 0.1:
        ri1 = np.pi * (BI * np.log((BI + BJ) / BI) + BJ * np.log((BI + BJ) / BJ))
        if G == 0.0:
            return ri1
        temp = 0.5 * BI ** 2 * np.log(BI / (BI + BJ)) + 0.5 * BJ ** 2 * np.log(
            BJ / (BI + BJ)
        )
        temp += 0.75 * (BI ** 2 + BJ ** 2) - 0.25 * (BI + BJ) ** 2
        temp -= 0.5 * (BI - BJ) ** 2
        ri1 += np.pi * G ** 2 * temp
        return ri1

    ri1 = (
        2.0 * 0.5772157
        + np.log(BI * G ** 2)
        + np.log(BJ * G ** 2)
        - 2.0 * expi(-BI * BJ * G ** 2 / (BI + BJ))
    )

    ri1 += RIH1(BI * G ** 2, BI * G ** 2 * BI / (BI + BJ), BI * G ** 2)

    ri1 += RIH1(BJ * G ** 2, BJ * G ** 2 * BJ / (BI + BJ), BJ * G ** 2)
    ri1 *= np.pi / G ** 2

    return ri1


def RI2(BI, BJ, G, U):
    # "ZWEITES INTEGRAL FUER DIE ABSORPTIONSPOTENTIALE"

    U2 = U ** 2
    U22 = 0.5 * U2
    G2 = G ** 2
    BIUH = BI + 0.5 * U2
    BJUH = BJ + 0.5 * U2
    BIU = BI + U2
    BJU = BJ + U2

    # "IST DIE ASYMPTOTISCHE ENTWICKLUNG ANWENDBAR?""
    EPS = np.max([BI, BJ, U2])
    EPS = EPS * G2

    if EPS <= 0.1:
        ri2 = (BI + U2) * np.log((BI + BJ + U2) / (BI + U2))
        ri2 = ri2 + BJ * np.log((BI + BJ + U2) / (BJ + U2))
        if U2 > 0.0:  # check if U=0, in which case we will force 0 * ∞ = 0
            ri2 = ri2 + U2 * np.log(U2 / (BJ + U2))
        ri2 = ri2 * np.pi
        if G == 0.0:
            return ri2
        if U2 > 0.0:
            TEMP = 0.5 * U22 * U22 * np.log(BIU * BJU / (U2 * U2))
        else:
            TEMP = 0.0
        TEMP = TEMP + 0.5 * BIUH * BIUH * np.log(BIU / (BIUH + BJUH))
        TEMP = TEMP + 0.5 * BJUH * BJUH * np.log(BJU / (BIUH + BJUH))
        TEMP = TEMP + 0.25 * BIU * BIU + 0.5 * BI * BI
        TEMP = TEMP + 0.25 * BJU * BJU + 0.5 * BJ * BJ
        TEMP = TEMP - 0.25 * (BIUH + BJUH) * (BIUH + BJUH)
        TEMP = TEMP - 0.5 * ((BI * BIU - BJ * BJU) / (BIUH + BJUH)) ** 2
        TEMP = TEMP - U22 * U22
        ri2 = ri2 + np.pi * G2 * TEMP
        return ri2

    ri2 = expi(-0.5 * U2 * G2 * BIUH / BIU) + expi(-0.5 * U2 * G2 * BJUH / BJU)
    ri2 = ri2 - expi(-BIUH * BJUH * G2 / (BIUH + BJUH)) - expi(-0.25 * U2 * G2)
    ri2 = 2.0 * ri2
    X1 = 0.5 * U2 * G2
    X2 = 0.25 * U2 * G2
    X3 = 0.25 * U2 * U2 * G2 / BIU
    ri2 = ri2 + RIH1(X1, X2, X3)

    X1 = 0.5 * U2 * G2
    X2 = 0.25 * U2 * G2
    X3 = 0.25 * U2 * U2 * G2 / BJU
    ri2 = ri2 + RIH1(X1, X2, X3)

    X1 = BIUH * G2
    X2 = BIUH * BIUH * G2 / (BIUH + BJUH)
    X3 = BIUH * BIUH * G2 / BIU
    ri2 = ri2 + RIH1(X1, X2, X3)

    X1 = BJUH * G2
    X2 = BJUH * BJUH * G2 / (BIUH + BJUH)
    X3 = BJUH * BJUH * G2 / BJU
    ri2 = ri2 + RIH1(X1, X2, X3)

    ri2 = ri2 * np.pi / G2

    return ri2


def RIH1(X1, X2, X3):
    # "WERTET DEN AUSDRUCK EXP(-X1) * ( EI(X2)-EI(X3) ) AUS"
    if (X2 <= 20.0) and (X3 <= 20.0):
        rih1 = np.exp(-X1) * (expi(X2) - expi(X3))
        return rih1
    if X2 > 20.0:
        rih1 = np.exp(X2 - X1) * RIH2(X2) / X2
    else:
        rih1 = np.exp(-X1) * expi(X2)

    if X3 > 20.0:
        rih1 = rih1 - np.exp(X3 - X1) * RIH2(X3) / X3
    else:
        rih1 = rih1 - np.exp(-X1) * expi(X3)

    return rih1


def RIH2(X):
    """
    WERTET X*EXP(-X)*EI(X) AUS FUER GROSSE X
    DURCH INTERPOLATION DER TABELLE ... AUS ABRAMOWITZ
    """
    idx = int(200.0 / X)
    return RIH2_tabulated_data[idx] + 200.0 * (
        RIH2_tabulated_data[idx + 1] - RIH2_tabulated_data[idx]
    ) * ((1.0 / X) - 0.5e-3 * idx)


def RIH3(X):
    # "WERTET DEN AUSDRUCK EXP(-X) * EI(X) AUS"
    if X <= 20.0:
        return np.exp(-X) * expi(X)
    else:
        return RIH2(X) / X


##################
# TABULATED DATA #
##################

# fmt:off

RIH2_tabulated_data = np.array([1.000000,1.005051,1.010206,1.015472,1.020852,
                                1.026355,1.031985,1.037751,1.043662,1.049726,
                                1.055956,1.062364,1.068965,1.075780,1.082830,
                                1.090140,1.097737,1.105647,1.113894,1.122497,
                                1.131470])


WK_A_param = np.array([
                    0.00427, 0.00957, 0.00802, 0.00209,
                    0.01217, 0.02616,-0.00884, 0.01841,
                    0.00251, 0.03576, 0.00988, 0.02370,
                    0.01596, 0.02959, 0.04024, 0.01001,
                    0.03652, 0.01140, 0.05677, 0.01506,
                    0.04102, 0.04911, 0.05296, 0.00061,
                    0.04123, 0.05740, 0.06529, 0.00373,
                    0.03547, 0.03133, 0.10865, 0.01615,
                    0.03957, 0.07225, 0.09581, 0.00792,
                    0.02597, 0.02197, 0.13762, 0.05394,
                    0.03283, 0.08858, 0.11688, 0.02516,
                    0.03833, 0.17124, 0.03649, 0.04134,
                    0.04388, 0.17743, 0.05047, 0.03957,
                    0.03812, 0.17833, 0.06280, 0.05605,
                    0.04166, 0.17817, 0.09479, 0.04463,
                    0.04003, 0.18346, 0.12218, 0.03753,
                    0.04245, 0.17645, 0.15814, 0.03011,
                    0.05011, 0.16667, 0.17074, 0.04358,
                    0.04058, 0.17582, 0.20943, 0.02922,
                    0.04001, 0.17416, 0.20986, 0.05497,
                    0.09685, 0.14777, 0.20981, 0.04852,
                    0.06667, 0.17356, 0.22710, 0.05957,
                    0.05118, 0.16791, 0.26700, 0.06476,
                    0.03204, 0.18460, 0.30764, 0.05052,
                    0.03866, 0.17782, 0.31329, 0.06898,
                    0.05455, 0.16660, 0.33208, 0.06947,
                    0.05942, 0.17472, 0.34423, 0.06828,
                    0.06049, 0.16600, 0.37302, 0.07109,
                    0.08034, 0.15838, 0.40116, 0.05467,
                    0.02948, 0.19200, 0.42222, 0.07480,
                    0.16157, 0.32976, 0.18964, 0.06148,
                    0.16184, 0.35705, 0.17618, 0.07133,
                    0.06190, 0.18452, 0.41600, 0.12793,
                    0.15913, 0.41583, 0.13385, 0.10549,
                    0.16514, 0.41202, 0.12900, 0.13209,
                    0.15798, 0.41181, 0.14254, 0.14987,
                    0.16535, 0.44674, 0.24245, 0.03161,
                    0.16039, 0.44470, 0.24661, 0.05840,
                    0.16619, 0.44376, 0.25613, 0.06797,
                    0.16794, 0.44505, 0.27188, 0.07313,
                    0.16552, 0.45008, 0.30474, 0.06161,
                    0.17327, 0.44679, 0.32441, 0.06143,
                    0.16424, 0.45046, 0.33749, 0.07766,
                    0.18750, 0.44919, 0.36323, 0.05388,
                    0.16081, 0.45211, 0.40343, 0.06140,
                    0.16599, 0.43951, 0.41478, 0.08142,
                    0.16547, 0.44658, 0.45401, 0.05959,
                    0.17154, 0.43689, 0.46392, 0.07725,
                    0.15752, 0.44821, 0.48186, 0.08596,
                    0.15732, 0.44563, 0.48507, 0.10948,
                    0.16971, 0.42742, 0.48779, 0.13653,
                    0.14927, 0.43729, 0.49444, 0.16440,
                    0.18053, 0.44724, 0.48163, 0.15995,
                    0.13141, 0.43855, 0.50035, 0.22299,
                    0.31397, 0.55648, 0.39828, 0.04852,
                    0.32756, 0.53927, 0.39830, 0.07607,
                    0.30887, 0.53804, 0.42265, 0.09559,
                    0.28398, 0.53568, 0.46662, 0.10282,
                    0.35160, 0.56889, 0.42010, 0.07246,
                    0.33810, 0.58035, 0.44442, 0.07413,
                    0.35449, 0.59626, 0.43868, 0.07152,
                    0.35559, 0.60598, 0.45165, 0.07168,
                    0.38379, 0.64088, 0.41710, 0.06708,
                    0.40352, 0.64303, 0.40488, 0.08137,
                    0.36838, 0.64761, 0.47222, 0.06854,
                    0.38514, 0.68422, 0.44359, 0.06775,
                    0.37280, 0.67528, 0.47337, 0.08320,
                    0.39335, 0.70093, 0.46774, 0.06658,
                    0.40587, 0.71223, 0.46598, 0.06847,
                    0.39728, 0.73368, 0.47795, 0.06759,
                    0.40697, 0.73576, 0.47481, 0.08291,
                    0.40122, 0.78861, 0.44658, 0.08799,
                    0.41127, 0.76965, 0.46563, 0.10180,
                    0.39978, 0.77171, 0.48541, 0.11540,
                    0.39130, 0.80752, 0.48702, 0.11041,
                    0.40436, 0.80701, 0.48445, 0.12438,
                    0.38816, 0.80163, 0.51922, 0.13514,
                    0.39551, 0.80409, 0.53365, 0.13485,
                    0.40850, 0.83052, 0.53325, 0.11978,
                    0.40092, 0.85415, 0.53346, 0.12747,
                    0.41872, 0.88168, 0.54551, 0.09404,
                    0.43358, 0.88007, 0.52966, 0.12059,
                    0.40858, 0.87837, 0.56392, 0.13698,
                    0.41637, 0.85094, 0.57749, 0.16700,
                    0.38951, 0.83297, 0.60557, 0.20770,
                    0.41677, 0.88094, 0.55170, 0.21029,
                    0.50089, 1.00860, 0.51420, 0.05996,
                    0.47470, 0.99363, 0.54721, 0.09206,
                    0.47810, 0.98385, 0.54905, 0.12055,
                    0.47903, 0.97455, 0.55883, 0.14309,
                    0.48351, 0.98292, 0.58877, 0.12425,
                    0.48664, 0.98057, 0.61483, 0.12136,
                    0.46078, 0.97139, 0.66506, 0.13012,
                    0.49148, 0.98583, 0.67674, 0.09725,
                    0.50865, 0.98574, 0.68109, 0.09977,
                    0.46259, 0.97882, 0.73056, 0.12723,
                    0.46221, 0.95749, 0.76259, 0.14086,
                    0.48500, 0.95602, 0.77234, 0.13374,
                    ]).reshape(98,4)


WK_B_param = np.array([
                  4.17218, 16.05892, 26.78365, 69.45643,
                  1.83008,  7.20225, 16.13585, 18.75551,
                  0.02620,  2.00907, 10.80597,130.49226,
                  0.38968,  1.99268, 46.86913,108.84167,
                  0.50627,  3.68297, 27.90586, 74.98296,
                  0.41335, 10.98289, 34.80286,177.19113,
                  0.29792,  7.84094, 22.58809, 72.59254,
                  0.17964,  2.60856, 11.79972, 38.02912,
                  0.16403,  3.96612, 12.43903, 40.05053,
                  0.09101,  0.41253,  5.02463, 17.52954,
                  0.06008,  2.07182,  7.64444,146.00952,
                  0.07424,  2.87177, 18.06729, 97.00854,
                  0.09086,  2.53252, 30.43883, 98.26737,
                  0.05396,  1.86461, 22.54263, 72.43144,
                  0.05564,  1.62500, 24.45354, 64.38264,
                  0.05214,  1.40793, 23.35691, 53.59676,
                  0.04643,  1.15677, 19.34091, 52.88785,
                  0.07991,  1.01436, 15.67109, 39.60819,
                  0.03352,  0.82984, 14.13679,200.97722,
                  0.02289,  0.71288, 11.18914,135.02390,
                  0.12527,  1.34248, 12.43524,131.71112,
                  0.05198,  0.86467, 10.59984,103.56776,
                  0.03786,  0.57160,  8.30305, 91.78068,
                  0.00240,  0.44931,  7.92251, 86.64058,
                  0.01836,  0.41203,  6.73736, 76.30466,
                  0.03947,  0.43294,  6.26864, 71.29470,
                  0.03962,  0.43253,  6.05175, 68.72437,
                  0.03558,  0.39976,  5.36660, 62.46894,
                  0.05475,  0.45736,  5.38252, 60.43276,
                  0.00137,  0.26535,  4.48040, 54.26088,
                  0.10455,  2.18391,  9.04125, 75.16958,
                  0.09890,  2.06856,  9.89926, 68.13783,
                  0.01642,  0.32542,  3.51888, 44.50604,
                  0.07669,  1.89297, 11.31554, 46.32082,
                  0.08199,  1.76568,  9.87254, 38.10640,
                  0.06939,  1.53446,  8.98025, 33.04365,
                  0.07044,  1.59236, 17.53592,215.26198,
                  0.06199,  1.41265, 14.33812,152.80257,
                  0.06364,  1.34205, 13.66551,125.72522,
                  0.06565,  1.25292, 13.09355,109.50252,
                  0.05921,  1.15624, 13.24924, 98.69958,
                  0.06162,  1.11236, 12.76149, 90.92026,
                  0.05081,  0.99771, 11.28925, 84.28943,
                  0.05120,  1.08672, 12.23172, 85.27316,
                  0.04662,  0.85252, 10.51121, 74.53949,
                  0.04933,  0.79381,  9.30944, 41.17414,
                  0.04481,  0.75608,  9.34354, 67.91975,
                  0.04867,  0.71518,  8.40595, 64.24400,
                  0.03672,  0.64379,  7.83687, 73.37281,
                  0.03308,  0.60931,  7.04977, 64.83582,
                  0.04023,  0.58192,  6.29247, 55.57061,
                  0.02842,  0.50687,  5.60835, 48.28004,
                  0.03830,  0.58340,  6.47550, 47.08820,
                  0.02097,  0.41007,  4.52105, 37.18178,
                  0.07813,  1.45053, 15.05933,199.48830,
                  0.08444,  1.40227, 13.12939,160.56676,
                  0.07206,  1.19585, 11.55866,127.31371,
                  0.05717,  0.98756,  9.95556,117.31874,
                  0.08249,  1.43427, 12.37363,150.55968,
                  0.07081,  1.31033, 11.44403,144.17706,
                  0.07442,  1.38680, 11.54391,143.72185,
                  0.07155,  1.34703, 11.00432,140.09138,
                  0.07794,  1.55042, 11.89283,142.79585,
                  0.08508,  1.60712, 11.45367,116.64063,
                  0.06520,  1.32571, 10.16884,134.69034,
                  0.06850,  1.43566, 10.57719,131.88972,
                  0.06264,  1.26756,  9.46411,107.50194,
                  0.06750,  1.35829,  9.76480,127.40374,
                  0.06958,  1.38750,  9.41888,122.10940,
                  0.06574,  1.31578,  9.13448,120.98209,
                  0.06517,  1.29452,  8.67569,100.34878,
                  0.06213,  1.30860,  9.18871, 91.20213,
                  0.06292,  1.23499,  8.42904, 77.59815,
                  0.05693,  1.15762,  7.83077, 67.14066,
                  0.05145,  1.11240,  8.33441, 65.71782,
                  0.05573,  1.11159,  8.00221, 57.35021,
                  0.04855,  0.99356,  7.38693, 51.75829,
                  0.04981,  0.97669,  7.38024, 44.52068,
                  0.05151,  1.00803,  8.03707, 45.01758,
                  0.04693,  0.98398,  7.83562, 46.51474,
                  0.05161,  1.02127,  9.18455, 64.88177,
                  0.05154,  1.03252,  8.49678, 58.79463,
                  0.04200,  0.90939,  7.71158, 57.79178,
                  0.04661,  0.87289,  6.84038, 51.36000,
                  0.04168,  0.73697,  5.86112, 43.78613,
                  0.04488,  0.83871,  6.44020, 43.51940,
                  0.05786,  1.20028, 13.85073,172.15909,
                  0.05239,  1.03225, 11.49796,143.12303,
                  0.05167,  0.98867, 10.52682,112.18267,
                  0.04931,  0.95698,  9.61135, 95.44649,
                  0.04748,  0.93369,  9.89867,102.06961,
                  0.04660,  0.89912,  9.69785,100.23434,
                  0.04323,  0.78798,  8.71624, 92.30811,
                  0.04641,  0.85867,  9.51157,111.02754,
                  0.04918,  0.87026,  9.41105,104.98576,
                  0.03904,  0.72797,  8.00506, 86.41747,
                  0.03969,  0.68167,  7.29607, 75.72682,
                  0.04291,  0.69956,  7.38554, 77.18528,
                  ]).reshape(98,4)
