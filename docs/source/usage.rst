Usage
=====

.. _installation:

Installation
------------

To use Nanoworks, first install the required system files (given for Debian/Ubuntu based distros):

.. code-block:: console

   $ sudo apt update && sudo apt upgrade -y
   $ sudo apt install -y python3-tk python3-venv python3-pip unzip python-is-python3 \
                    python3-dev libopenblas-dev libxc-dev libscalapack-mpi-dev \
                    libfftw3-dev libkim-api-dev openkim-models libkim-api2 pkg-config \
                    task-spooler

Then if you do not have a python environment, create one and activate it:

.. code-block:: console

   $ python -m venv ~/nwenv
   $ source ~/nwenv/bin/activate

Then install nanoworks wit pip

.. code-block:: console
   
   (nwenv) $ pip install lumache
