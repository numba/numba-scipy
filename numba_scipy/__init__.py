
from . import _version
__version__ = _version.get_versions()['version']


def _init_extension():
    '''Register SciPy functions with Numba.

    This entry_point is called by Numba when it initializes.
    '''
    from . import special, sparse
