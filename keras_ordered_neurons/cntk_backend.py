import cntk as C
import numpy as np
from .backend import backend as K


__all__ = ['cumsum']


def cumsum(x, axis=-1):
    if axis != -1 and axis != K.ndim(x) - 1:
        raise ValueError('Only the last axis could be used, found: {}'.format(axis))
    dim = x.shape[-1]
    U = C.constant(np.triu(np.ones((dim, dim))).astype(x.dtype))
    out = C.times(x, U)
    return out
