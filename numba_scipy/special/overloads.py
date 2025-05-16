import logging
import numba
import scipy.special as sc

from . import signatures

logger = logging.getLogger("numba_scipy.special.overloads")

def choose_kernel(name, all_signatures):

    def choice_function(*args):
        logger.debug(f"Choosing kernel for function '{name}' with args types: {args}")
        for signature in all_signatures:
            if args == signature:
                logger.info(f"Found matching signature {signature} for function '{name}'")
                f = signatures.name_and_types_to_pointer[(name, *signature)]
                if f is None:
                    logger.error(f"No function pointer found for {name} with signature {signature}")
                    raise RuntimeError(f"Missing function pointer for {name} with signature {signature}")
                return lambda *args: f(*args)
        logger.warning(f"No matching signature found for {name} with args types {args}")
        # Optional fallback or raise error here if no match found
        raise TypeError(f"No matching kernel found for {name} with args {args}")

    return choice_function


def add_overloads():
    logger.info("Adding overloads for scipy.special functions")
    for name, all_signatures in signatures.name_to_numba_signatures.items():
        logger.debug(f"Adding overload for function: {name} with signatures: {all_signatures}")
        try:
            sc_function = getattr(sc, name)
        except AttributeError:
            logger.warning(f"scipy.special has no function named '{name}', skipping")
            continue
        numba.extending.overload(sc_function)(
            choose_kernel(name, all_signatures)
        )
    logger.info("Finished adding overloads")
