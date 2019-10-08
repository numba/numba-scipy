#!/bin/bash

set -v -e

# first configure conda to have more tolerance of network problems, these
# numbers are not scientifically chosen, just merely larger than defaults
conda config --write-default
conda config --set remote_connect_timeout_secs 30.15
conda config --set remote_max_retries 10
conda config --set remote_read_timeout_secs 120.2
if [[ $(uname) == Linux ]]; then
    if [[ "$CONDA_SUBDIR" != "linux-32" && "$BITS32" != "yes" ]] ; then
        conda config --set restore_free_channel true
    fi
fi
conda info
conda config --show

CONDA_INSTALL="conda install -q -y"
PIP_INSTALL="pip install -q"

# Deactivate any environment
source deactivate
# Display root environment (for debugging)
conda list
# Clean up any left-over from a previous build
# (note workaround for https://github.com/conda/conda/issues/2679:
#  `conda env remove` issue)
conda remove --all -q -y -n $CONDA_ENV

# Create a base env
conda create -n $CONDA_ENV -q -y python=$PYTHON numpy=$NUMPY scipy=$SCIPY pip

# Activate
set +v
source activate $CONDA_ENV
set -v

# 32bit linux needs the numba channel to get a conda package as the distro
# channels stopped shipping for 32bit linux packages, this branching is
# superfluous but sets up for adding later conditional package installation.
if [[ $(uname) == Linux ]]; then
    if [[ "$CONDA_SUBDIR" == "linux-32" || "$BITS32" == "yes" ]] ; then
        $CONDA_INSTALL -c numba numba
        # Work around https://github.com/pytest-dev/pytest/issues/3280
        $CONDA_INSTALL pytest attrs==19.1.0
    else
        $CONDA_INSTALL numba pytest
    fi
elif  [[ $(uname) == Darwin ]]; then
    $CONDA_INSTALL numba pytest
fi

# environment dump for debug
echo "-------------------------------------------------------------------------"
conda env export
echo "-------------------------------------------------------------------------"
