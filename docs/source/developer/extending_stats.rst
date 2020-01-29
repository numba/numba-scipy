Extending numba-scipy.stats
===========================

Contributions to numba-scipy stats are always welcomed! Even simple documentation
improvements are encouraged.  If you have questions, don't hesitate to ask them.

The following is a description with the most common ways in which numba-scipy stats can be extended.


Internal structure
-------------------

Numba-scipy tries to follow `scipy.stats <https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html/>`_ as closely as possible, to make it easier for users to understand and extend
the internal structure.

Each statistical distribution is represented by one class, for example
``numba_scipy.stats._continuous_distns.norm_gen_jit``. Following SciPy's convention (and breaking with Python's) these
classes have lowercase names. These classes can be thought of as generators or generic distributions,
since the actual distributions---those with a concrete value for their parameters---are the instances of these classes.
This is why in SciPy one uses, for example, ``stats.norm`` and not ``stats.norm_gen``. In SciPy, ``stats.norm``
is an instance of ``stats.norm_gen`` with the default value ``loc=0`` and ``scale=1`` for its parameters.


In both numba-scipy and SciPy, each of these generator classes inherits from a parent class, ``rv_continuous`` or ``rv_discrete`` for continuous and
discrete distributions respectively. In turn, these inherit from ``rv_generic``. These classes contain the public
methods of all distributions, for example

- rvs: Random Variates

- pdf: Probability Density Function

- cdf: Cumulative Distribution Function

- sf: Survival Function (1-CDF)

- ppf: Percent Point Function (Inverse of CDF)

- isf: Inverse Survival Function (Inverse of SF)

- stats: Return mean, variance, (Fisher’s) skew, or (Fisher’s) kurtosis

- moment: non-central moments of the distribution.

These public methods will then call a private method in each individual distribution class that implements the
specific calculations for each distribution.

There are three main areas in which numba-scipy stats can be extended:

- to add a new private method on an existing distribution
- to add a new public method for the base classes
- to add a new distribution.

.. StabilityWarning::
   numba-scipy is currently in an early stage, and its internal structure---including the descriptions below---are
   subject to change.

Adding a new private method
---------------------------

In this case, the public method in the base class already exists, but the distribution of interest has not implemented
it yet. If, for example, this were the existing class::

    @jitclass(spec=rv_continuous_spec + [])
    class norm_gen_jit(rv_continuous):
        def _rvs(self, size):
            return np.random.standard_normal(size)

then it could be extended with a method returning its probability density function::


    @jitclass(spec=rv_continuous_spec + [])
    class norm_gen_jit(rv_continuous):
        def _rvs(self, size):
            return np.random.standard_normal(size)
        def _pdf(self, x):
            # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
            return _norm_pdf(x)



This method is, however, private and it will be called by ``pdf``,  which is the public method
implemented in ``rv_continuous``.
In case ``rv_continuous`` did not have ``pdf`` already implemented, then it would be necessary to do so.

Adding a new public method
---------------------------

Continuing with our example, if ``rv_continous`` did not implement ``pdf`` yet::

    class rv_continuous(rv_generic):
        def __init__(self, name=None, seed=None):
            ...

then it would be necessary to add it. A good starting point is to look into SciPy stats' implementation and bring
as much as possible. It is expected that it is not possible to jit all functionality from SciPy. For example,
``**kwargs`` cannot be used in a ``jitclass`` so the signature would have to be different in numba-scipy than in SciPy::

    class rv_continuous(rv_generic):
        def __init__(self, name=None, seed=None):
            ...

        def pdf(self, x, *args):
        ...



Adding a new distribution
---------------------------
Our final example concerns the addition of a new class. In this case, a new class would have to be created in either
``_continuous_distns.py`` or ``_discrete_distns.py``. The required private methods can then be implemented for this
class::

    @jitclass(spec=rv_continuous_spec + [])
    class norm_gen_jit(rv_continuous):
        def _rvs(self, size):
            return np.random.standard_normal(size)
