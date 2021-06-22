

def cartesian_aberrations(qx, qy, lam, C):
    """
    Args:
        qx:
        qy:
        lam:
        C:
    """

    u = qx * lam
    v = qy * lam
    u2 = u ** 2
    u3 = u ** 3
    u4 = u ** 4

    v2 = v ** 2
    v3 = v ** 3
    v4 = v ** 4

    aberr = Param()
    aberr.C1 = C[0]
    aberr.C12a = C[1]
    aberr.C12b = C[2]
    aberr.C21a = C[3]
    aberr.C21b = C[4]
    aberr.C23a = C[5]
    aberr.C23b = C[6]
    aberr.C3 = C[7]
    aberr.C32a = C[8]
    aberr.C32b = C[9]
    aberr.C34a = C[10]
    aberr.C34b = C[11]

    chi = 1 / 2 * aberr.C1 * (u2 + v2)
    + 1 / 2 * (aberr.C12a * (u2 - v2) + 2 * aberr.C12b * u * v)
    + 1 / 3 * (aberr.C23a * (u3 - 3 * u * v2) + aberr.C23b * (3 * u2 * v - v3))
    + 1 / 3 * (aberr.C21a * (u3 + u * v2) + aberr.C21b * (v3 + u2 * v))
    + 1 / 4 * aberr.C3 * (u4 + v4 + 2 * u2 * v2)
    + 1 / 4 * aberr.C34a * (u4 - 6 * u2 * v2 + v4)
    + 1 / 4 * aberr.C34b * (4 * u3 * v - 4 * u * v3)
    + 1 / 4 * aberr.C32a * (u4 - v4)
    + 1 / 4 * aberr.C32b * (2 * u3 * v + 2 * u * v3)

    chi *= 2 * np.pi / lam

    return chi
