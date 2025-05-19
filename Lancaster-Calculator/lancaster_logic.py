import math

def lancaster_compression(z):
    if z == 0:
        return 0
    return math.log(z + 1) * math.exp(-1 / z)

def compute_codex_psi(chi, C, S):
    base = chi * C * (S ** 2)
    psi = lancaster_compression(base)
    return round(psi, 5)
