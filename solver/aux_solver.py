import arrayfire as af
import numpy as np
import os

def fftfreq(n, d, dtype='d'):
    if n % 2 == 0:
        return af.join(0, af.data.range(n // 2, 1, 1, dtype=dtype), -1.0 * (af.flip(af.range(n // 2, 1, 1, dtype=dtype), 0) + 1.0)) / (n * d)
    else:
        return af.join(0, af.data.range((n - 1) // 2 + 1, 1, 1, dtype=dtype), -1.0 * (af.flip(af.range((n - 1) // 2, 1, 1, dtype=dtype), 0) + 1.0)) / (n * d)

def kx_nd(kx, dimens):
    kx_d = af.tile(af.moddims(kx, kx.elements(), 1, 1, 1), 1, dimens[1], 1)
    return kx_d

def ky_nd(ky, dimens):
    ky_d = af.tile(af.moddims(ky, 1, ky.elements(), 1, 1), dimens[0], 1, 1)
    return ky_d

def kt_nd(kt, dimens):
    kt_d = af.tile(af.moddims(kt, 1, 1, kt.elements(), 1), dimens[0], dimens[1], 1)
    return kt_d

def k2_nd(dx, dy, dz, dimens, dtype):
    kx = kx_nd(2.0 * np.pi * fftfreq(dimens[0], dx, dtype), dimens)
    ky = ky_nd(2.0 * np.pi * fftfreq(dimens[1], dy, dtype), dimens)
    k2 = af.pow(kx, 2) + af.pow(ky, 2) 
    return k2

def k_nd(dx, dy, dz, dimens, dtype):
    kx = kx_nd(2.0 * np.pi * fftfreq(dimens[0], dx, dtype), dimens)
    ky = ky_nd(2.0 * np.pi* fftfreq(dimens[1], dy, dtype), dimens)
    k = kx + ky 
    return k

def create_folder(path):
    os.makedirs(path, exist_ok=True)
