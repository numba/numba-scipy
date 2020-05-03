import ctypes

from numba.extending import get_cython_function_address


def get_special_function_pointer(ctypes_signature, function_name):
    try:
        address = get_cython_function_address(
            'scipy.special.cython_special',
            function_name,
        )
    except ValueError:
        return None
    return ctypes.CFUNCTYPE(*ctypes_signature)(address)
