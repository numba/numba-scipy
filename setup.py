from setuptools import setup
import versioneer

metadata = dict(
    name='numba_scipy',
    description="numba_scipy extends Numba to make it aware of SciPy",
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
    url="https://github.com/numba/numba_scipy",
    download_url="https://github.com/numba/numba_scipy",
    packages=['numba_scipy'],
    setup_requires=[],
    install_requires=[],
    license="BSD",
    zip_safe=False,
    )

with open('README.rst') as f:
    metadata['long_description'] = f.read()

setup(**metadata)
