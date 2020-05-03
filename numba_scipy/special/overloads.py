import numba
import scipy.special as sc

from . import signatures


def choose_kernel(name, all_signatures):

    def choice_function(*args):
        for signature in all_signatures:
            if args == signature:
                f = signatures.name_and_types_to_pointer[(name, *signature)]
                if f is None:
                    # This could be a version of SciPy where this
                    # signature isn't present.
                    raise RuntimeError(
                        "Couldn't find signature for {} with arguments {}"
                        .format(name, args)
                    )
                return lambda *args: f(*args)

    return choice_function


def add_overloads():
    for name, all_signatures in signatures.name_to_numba_signatures.items():
        sc_function = getattr(sc, name, None)
        if sc_function is None:
            # This could be a version of SciPy where this function
            # isn't present.
            continue
        numba.extending.overload(sc_function)(
            choose_kernel(name, all_signatures)
        )
