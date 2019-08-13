import numba
import numpy as np
import scipy.special as sc

from . import signatures
from . import types


def choose_kernel(name, all_signatures):

    def choice_function(*args):
        for signature in all_signatures:
            print(signature)
            if args == signature:
                f = signatures.name_and_types_to_pointer[(name, signature)]
                return lambda *args: f(*args)

    return choice_function


def add_overloads():
    for name, all_signatures in signatures.name_to_numba_signatures.items():
        sc_function = getattr(sc, name)
        numba.extending.overload(sc_function)(
            choose_kernel(name, all_signatures)
        )
