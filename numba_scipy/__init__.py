
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def _init_extension():
    '''Register SciPy functions with Numba.

    This entry_point is called by Numba when it initializes.
    '''
    from . import special
