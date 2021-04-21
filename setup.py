from setuptools import setup, find_packages
import versioneer


_install_requires = ['scipy>=0.16,<=1.6.2', 'numba>=0.45']


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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Compilers",
    ],
    package_data={},
    scripts=[],
    author="Anaconda, Inc.",
    author_email="numba-users@continuum.io",
    url="https://github.com/numba/numba-scipy",
    download_url="https://github.com/numba/numba-scipy",
    packages=find_packages(),
    setup_requires=[],
    install_requires=_install_requires,
    entry_points={
        "numba_extensions": [
            "init = numba_scipy:_init_extension",
        ],
    },
    license="BSD",
    zip_safe=False,
)


with open('README.rst') as f:
    metadata['long_description'] = f.read()


setup(**metadata)
