"""
==========================================
Statistical functions (:mod:`numba-scipy.stats`)
==========================================

.. currentmodule:: numba-scipy.stats

This module aims to provide the existing functionality in SciPy's stats_ module
This module contains a large number of probability distributions as
well as a growing library of statistical functions.

Each univariate distribution is an instance of a subclass of `rv_continuous`
(`rv_discrete` for discrete distributions):

.. _stats: https://docs.scipy.org/doc/scipy/reference/stats.html

.. autosummary::
   :toctree: generated/

   rv_continuous
   rv_discrete
   rv_histogram

Continuous distributions
========================

.. autosummary::
   :toctree: generated/

   norm              -- Normal (Gaussian)

Multivariate distributions
==========================

.. autosummary::
   :toctree: generated/


Discrete distributions
======================

.. autosummary::
   :toctree: generated/


An overview of statistical functions is given below.


Summary statistics
==================

.. autosummary::
   :toctree: generated/



Frequency statistics
====================


Correlation functions
=====================


Statistical tests
=================



Transformations
===============



Statistical distances
=====================



Random variate generation
=========================


Circular statistical functions
==============================



Contingency table functions
===========================


Plot-tests
==========



Masked statistics functions
===========================



Univariate and multivariate kernel density estimation
=====================================================



Warnings used in :mod:`scipy.stats`
===================================



"""

def register_overloads():
    from ._continuous_distns import register_overloads as cont_reg_overloads
    cont_reg_overloads()

# from .distributions import *
#
# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
#
#
