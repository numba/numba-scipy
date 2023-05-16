from textwrap import dedent

import numba
import scipy.special as sc

from . import signatures


def choose_kernel(name, all_signatures):

    def choice_function(*args):
        scalar_args = ()
        has_arrays = False
        for a in args:
            if isinstance(a, numba.types.Array):
                scalar_args += (a.dtype,)
                has_arrays = True
            else:
                scalar_args += (a,)

        for signature in all_signatures:
            if scalar_args == signature:
                f = signatures.name_and_types_to_pointer[(name, *signature)]

                if has_arrays:

                    args_str = ", ".join([f"arg{i}" for i in range(len(args))])

                    global_env = {"f": f, "numba": numba}

                    vectorized_fn_src = dedent(f"""
                @numba.vectorize
                def f_vec({args_str}):
                    return f({args_str})
                    """)

                    mod_code = compile(vectorized_fn_src, "<meta>", mode="exec")
                    local_env = {}
                    exec(mod_code, global_env, local_env)

                    f_vec = local_env["f_vec"]

                    def f_res(*args):
                        return f_vec(*args)

                else:

                    def f_res(*args):
                        return f(*args)

                return f_res

    return choice_function


def add_overloads():
    for name, all_signatures in signatures.name_to_numba_signatures.items():
        sc_function = getattr(sc, name)
        numba.extending.overload(sc_function)(
            choose_kernel(name, all_signatures)
        )
