import os
import re
import collections

import scipy.special.cython_special as cysc

SPECIAL_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'numba_scipy',
    'special'
)

SIGNATURES_TEMPLATE = """\
import ctypes

import numba
from numba.extending import get_cython_function_address

from .types import numba_long

name_to_numba_signatures = {{
    {NAME_TO_NUMBA_SIGNATURES}
}}

name_and_types_to_pointer = {{
    {NAME_AND_TYPES_TO_POINTER}
}}
"""

CYTHON_TO_NUMBA = {
    'double': 'numba.types.float64',
    'float': 'numba.types.float32',
    'long': 'numba_long',
}

NUMBA_TO_CTYPES = {
    'numba.types.float64': 'ctypes.c_double',
    'numba.types.float32': 'ctypes.c_float',
    'numba_long': 'ctypes.c_long',
}


def parse_capsule_name(capsule):
    # There isn't a Python equivalent to `PyCapsule_GetName`, so
    # resort to a hacky method for finding the signature.
    match = re.match(
        '\<capsule object "(?P<signature>.+)" at 0x[a-f0-9]+\>',
        str(capsule),
    )
    if match is None:
        raise ValueError('Unexpected capsule name {}'.format(capsule))

    signature = match.group('signature')
    match = re.match('(?P<return_type>.+) \((?P<arg_types>.+)\)', signature)
    if match is None:
        raise ValueError('Unexpected signature {}'.format(signature))

    args = [
        arg_type for arg_type in match.group('arg_types').split(', ')
        if arg_type != 'int __pyx_skip_dispatch'
    ]
    return [match.group('return_type')] + args


def de_mangle_function_name(mangled_name):
    match = re.match('(__pyx_fuse(_[0-9])*)?(?P<name>.+)', mangled_name)
    if match is None:
        raise ValueError('Unexpected mangled name {}'.format(mangled_name))

    return match.group('name')


def get_signatures_from_pyx_capi():
    signature_to_pointer = {}

    for mangled_name, capsule in cysc.__pyx_capi__.items():
        signature_types = parse_capsule_name(capsule)
        numba_signature = [
            CYTHON_TO_NUMBA.get(t) for t in parse_capsule_name(capsule)
        ]
        if any(t is None for t in numba_signature):
            # We don't know how to handle this kernel yet.
            continue

        signature_to_pointer[(mangled_name, *numba_signature)] = capsule

    return signature_to_pointer


def generate_signatures_file(signature_to_pointer):
    name_to_numba_signatures = collections.defaultdict(list)
    name_and_types_to_pointer = []
    for mangled_name, *signature in signature_to_pointer.keys():
        name = de_mangle_function_name(mangled_name)
        name_to_numba_signatures[name].append(
            '({},)'.format(', '.join(signature[1:]))
        )

        key = "('{}', {})".format(name, ', '.join(signature[1:]))
        address = (
            "get_cython_function_address('scipy.special.cython_special', '{}')"
            .format(mangled_name)
        )
        ctypes_signature = [NUMBA_TO_CTYPES[t] for t in signature]
        ctypes_cast = (
            'ctypes.CFUNCTYPE({})'.format(', '.join(ctypes_signature))
        )
        name_and_types_to_pointer.append(
            '{}: {}({})'.format(key, ctypes_cast, address)
        )

    name_to_numba_signatures = [
        "'{}': [{}]".format(name, ', '.join(signatures))
        for name, signatures in name_to_numba_signatures.items()
    ]
    name_to_numba_signatures = ',\n    '.join(name_to_numba_signatures)

    name_and_types_to_pointer = ',\n    '.join(name_and_types_to_pointer)
    content = SIGNATURES_TEMPLATE.format(
        NAME_TO_NUMBA_SIGNATURES=name_to_numba_signatures,
        NAME_AND_TYPES_TO_POINTER=name_and_types_to_pointer
    )
    with open(os.path.join(SPECIAL_DIR, 'signatures.py'), 'w') as f:
        f.write(content)


def main():
    signature_to_pointer = get_signatures_from_pyx_capi()
    generate_signatures_file(signature_to_pointer)


if __name__ == '__main__':
    main()
