import cntk as C
import numpy as np


__all__ = ['cumsum']


def cumsum(x, axis=0):
    dim = x.shape[axis]
    U = C.constant(np.triu(np.ones((dim, dim))).astype(x.dtype))
    if axis != -1:
        x = C.swapaxes(x, -1, axis)
    out = C.times(x, U)
    if axis != -1:
        out = C.swapaxes(out, -1, axis)
    return out
