
Contributing to numba-scipy
===========================

Contributions to numba-scipy are always welcomed! Even simple documentation
improvements are encouraged.  If you have questions, don't hesitate to ask them
(see below).


Communication
-------------

Contact
'''''''

The Numba community uses Discourse for asking questions and having discussions
about numba-scipy. There are various categories available and it can be reached
at: `numba.discourse.group <https://numba.discourse.group/>`_. There is also a
category for `numba-scipy <https://numba.discourse.group/c/numba/numba-scipy>`_.

Real-time Chat
''''''''''''''

numba-scipy uses Gitter for public real-time chat.  To help improve the
signal-to-noise ratio, there are two channels:

* `numba/numba <https://gitter.im/numba/numba>`_: General discussion, questions,
  and debugging help.
* `numba/numba-dev <https://gitter.im/numba/numba-dev>`_: Discussion of PRs,
  planning, release coordination, etc.

Both channels are public.

Note that the Github issue tracker is the best place to report bugs.  Bug
reports in chat are difficult to track and likely to be lost.

.. _report-bugs:

Bug tracker
''''''''''''

The `Github issue tracker <https://github.com/numba/numba-scipy/issues>`_ is
used to track both bug reports and feature requests.

Getting set up
--------------

If you want to contribute, it's best to fork the `Github repository
<https://github.com/numba/numba-scipy>`_, then create a branch representing
your work.  When your work is ready, you should submit it as a pull
request from the Github interface.

If you want, you can submit a pull request even when you haven't finished
working.  This can be useful to gather feedback, or to stress your changes
against the :ref:`continuous integration <azure_ci>` platform.  In this
case, please prepend ``[WIP]`` to your pull request's title.

.. _buildenv:

Build environment
'''''''''''''''''

numba-scipy has a number of dependencies (mostly `Numba <http://http://numba.pydata.org/>`_ 
and `SciPy <https://www.scipy.org/scipylib/index.html>`_).  Unless you want to
build those dependencies yourself, it's recommended you use
`conda <http://conda.pydata.org/miniconda.html>`_ to create a dedicated
development environment and install pre-compiled versions of those dependencies
there.

First add the Anaconda Cloud ``numba`` channel so as to get development builds
of the numba library::

   $ conda config --add channels numba

Then create an environment with the right dependencies::

   $ conda create -n numba-scipy python=3.7 scipy numba

.. note::
   This installs an environment based on Python 3.7, but you can of course
   choose another version supported by Numba.

To activate the environment for the current shell session::

   $ conda activate numba-scipy

.. note::
   These instructions are for a standard Linux shell.  You may need to
   adapt them for other platforms.

Once the environment is activated, you have a dedicated Python with the
required dependencies.


Building numba-scipy
''''''''''''''''''''

For a convenient development workflow, it's recommended that you build
numba-scipy inside its source checkout::

   $ git clone git://github.com/numba/numba-scipy.git
   $ cd numba-scipy
   $ python setup.py develop


Running tests
'''''''''''''

numba-scipy is validated using a test suite comprised of various kind of tests
(unit tests, functional tests). The test suite is written using the
standard :py:mod:`unittest` framework and rely on ``pytest`` for execution. The
``pytest`` package will need installing to run the tests, using ``conda`` this
can be achieved by::

    $ conda install pytest

The tests can then be executed via ``python -m pytest``.


Development rules
-----------------

Code reviews
''''''''''''

Any non-trivial change should go through a code review by one or several of
the core developers.  The recommended process is to submit a pull request
on github.

A code review should try to assess the following criteria:

* general design and correctness
* code structure and maintainability
* coding conventions
* docstrings, comments
* test coverage

Coding conventions
''''''''''''''''''

All Python code should follow :pep:`8`. Code and documentation should generally
fit within 80 columns, for maximum readability with all existing tools (such as
code review UIs).

Stability
'''''''''

The repository's ``master`` branch is expected to be stable at all times.
This translates into the fact that the test suite passes without errors
on all supported platforms (see below).  This also means that a pull request
also needs to pass the test suite before it is merged in.

.. _azure_ci:

Platform support
''''''''''''''''

Every commit to the master branch is automatically tested on a selection of
platforms. `Azure <https://dev.azure.com/numba/numba/_build>`_ is used to to
provide public continuous integration information for as many combinations as
can be supported by the service. If you see problems on platforms with which you
are unfamiliar, feel free to ask for help in your pull request.  The numba-scipy
core developers can help diagnose cross-platform compatibility issues.


Documentation
''''''''''''''''''

This documentation is under the ``docs`` directory of the
`numba-scipy repository <https://github.com/numba/numba-scipy>`_.
It is built with `Sphinx <http://sphinx-doc.org/>`_, which is available
using conda or pip.

To build the documentation, you need the bootstrap theme::

   $ pip install sphinx_bootstrap_theme

You can edit the source files under ``docs/source/``, after which you can
build and check the documentation::

   $ make html
   $ open _build/html/index.html
