import os
import json
import collections
import ctypes

SPECIAL_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'numba_scipy',
    'special'
)

FUNCTION_POINTERS_TEMPLATE = """\
#cython: language_level=3
from cpython.long cimport PyLong_FromVoidPtr
import ctypes

import numba
cimport scipy.special.cython_special as sc

from .types import numba_long

pointers = {{
{POINTERS}
}}
"""

SIGNATURES_TEMPLATE = """\
import ctypes

import numba
import numpy as np

from .types import numba_long
from .function_pointers import pointers

name_to_numba_signatures = {{
{NAME_TO_NUMBA_SIGNATURES}
}}

name_and_types_to_pointer = {{
{NAME_AND_TYPES_TO_POINTER}
}}
"""

CTYPES_TO_NUMBA = {
    'c_double': 'numba.types.float64',
    'c_float': 'numba.types.float32',
    'c_long': 'numba_long'
}


def get_signatures():
    with open(os.path.join(SPECIAL_DIR, 'signatures.json')) as f:
        signatures = json.load(f)
    return signatures


def get_cython_specialized_name(name, cython_key):
    if cython_key == '':
        # Not a fused type function, so no need to specialize.
        return name

    cython_key = ', '.join(cython_key.split('|'))
    return '{}[{}]'.format(name, cython_key)


def get_ctypes_string_signature(ctypes_signature):
    joined_ctypes_signature = ', '.join(
        ['ctypes.{}'.format(t) for t in ctypes_signature]
    )
    return 'ctypes.CFUNCTYPE({})'.format(joined_ctypes_signature)


def get_dictionary_key(name, ctypes_signature, string_types=False):
    numba_types = [
        CTYPES_TO_NUMBA[arg] for arg in ctypes_signature[1:]
    ]
    if string_types:
        # Cython takes a long time to compile the dict when using
        # actual Numba types, so for compiled code use strings
        # instead.
        numba_types = [
            "'{}'".format(numba_type) for numba_type in numba_types
        ]

    return "('{}', ({},))".format(name, ', '.join(numba_types))


def generate_function_pointers(signatures):
    pointers = []
    for name, specializations in signatures.items():
        for cython_key, ctypes_signature in specializations.items():
            specialized_name = get_cython_specialized_name(name, cython_key)
            key = get_dictionary_key(name, ctypes_signature, string_types=True)

            cast = 'PyLong_FromVoidPtr(<void *>sc.{})'.format(
                specialized_name
            )
            pointers.append(
                '{}: {}'.format(key, cast)
            )

    pointers = '    ' + ',\n    '.join(pointers)
    content = FUNCTION_POINTERS_TEMPLATE.format(POINTERS=pointers)
    with open(os.path.join(SPECIAL_DIR, 'function_pointers.pyx'), 'w') as f:
        f.write(content)


def generate_name_to_signatures_map(signatures):
    name_to_numba_signatures = collections.defaultdict(list)
    name_and_types_to_pointer = []

    for name, specializations in signatures.items():
        for cython_key, ctypes_signature in specializations.items():
            key = get_dictionary_key(name, ctypes_signature)
            string_key = get_dictionary_key(
                name, ctypes_signature, string_types=True
            )

            # Add an entry that maps the function name to it's signatures
            numba_args = ', '.join(
                CTYPES_TO_NUMBA[arg] for arg in ctypes_signature[1:]
            )
            name_to_numba_signatures[name].append('({},)'.format(numba_args))

            # Add an entry that maps a name and specialization to the
            # function pointer.
            ctypes_string_signature = get_ctypes_string_signature(
                ctypes_signature
            )
            name_and_types_to_pointer.append(
                '{}: {}(pointers[{}])'.format(
                    key, ctypes_string_signature, string_key
                )
            )

    signatures_for_name = []
    for name, signatures in name_to_numba_signatures.items():
        signatures_for_name.append("'{}': [{}]".format(
            name, ', '.join(signatures)
        ))
    name_to_numba_signatures_str = '    ' + ',\n    '.join(signatures_for_name)
    name_and_types_to_pointer = ',\n    '.join(name_and_types_to_pointer)
    content = SIGNATURES_TEMPLATE.format(
        NAME_TO_NUMBA_SIGNATURES=name_to_numba_signatures_str,
        NAME_AND_TYPES_TO_POINTER=name_and_types_to_pointer
    )
    with open(os.path.join(SPECIAL_DIR, 'signatures.py'), 'w') as f:
        f.write(content)


def main():
    signatures = get_signatures()
    generate_function_pointers(signatures)
    generate_name_to_signatures_map(signatures)


if __name__ == '__main__':
    main()
