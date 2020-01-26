# -*- coding: utf-8 -*-

from ._distn_infrastructure import (get_distribution_names, rv_continuous,
                                    rv_continuous_spec)
from .utils import overload_pyclass
import scipy.stats as scipy_stats
from numba.extending import overload

from numba import jitclass
import numpy as np


@jitclass(spec=rv_continuous_spec + [])
class norm_gen_jit(rv_continuous):
    r"""A normal continuous random variable.
    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation.
    %(before_notes)s
    Notes
    -----
    The probability density function for `norm` is:
    .. math::
        f(x) = \frac{\exp(-x^2/2)}{\sqrt{2\pi}}
    for a real number :math:`x`.
    %(after_notes)s
    %(example)s
    """
    def _rvs(self, size):
        return np.random.standard_normal(size)



def register_overloads():
    overload_pyclass(scipy_stats._continuous_distns.norm_gen, norm_gen_jit)

# Collect names of classes and objects in this module.
# Temporarily disabled due to Numba issue #?? preventing the recognition of
# distribution instances

# pairs = list(globals().items())

# _distn_names, _distn_gen_names = get_distribution_names(pairs, rv_continuous)

# __all__ = _distn_names + _distn_gen_names

_distn_names = ['norm']
_distn_gen_names = [norm_gen_jit]