import os
from setuptools import setup, Extension
import versioneer


def get_extension_modules():
    try:
        from Cython.Build import cythonize
    except ImportError:
        return [
            Extension(
                'numba_scipy.special.function_pointers',
                [os.path.join('numba_scipy', 'special', 'function_pointers.c')]
            )
        ]

    return cythonize([
        Extension(
            'numba_scipy.special.function_pointers',
            [os.path.join('numba_scipy', 'special', 'function_pointers.pyx')]
        )
    ])


_install_requires = ['scipy>=0.16', 'numba>=0.45']

metadata = dict(
    name='numba-scipy',
    description="numba-scipy extends Numba to make it aware of SciPy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Compilers",
    ],
    package_data={},
    scripts=[],
    author="Anaconda, Inc.",
    author_email="numba-users@continuum.io",
    url="https://github.com/numba/numba-scipy",
    download_url="https://github.com/numba/numba-scipy",
    packages=['numba_scipy'],
    setup_requires=[],
    install_requires=_install_requires,
    ext_modules=get_extension_modules(),
    license="BSD",
    zip_safe=False,
    )

with open('README.rst') as f:
    metadata['long_description'] = f.read()

setup(**metadata)
