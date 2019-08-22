import ctypes

import numba

bytes_to_int_type = {
    2: numba.types.int16,
    4: numba.types.int32,
    8: numba.types.int64,
}

numba_long = bytes_to_int_type[ctypes.sizeof(ctypes.c_long)]
