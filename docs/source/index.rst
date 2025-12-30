Nanoworks
=========

`Nanoworks <https://nanoworks.readthedocs.io/>`_ is a powerful and user-friendly user interface (UI) tool for conducting Density Functional Theory (DFT) and molecular dynamics (MD) calculations. In the near future, machine learning features will be added.

**gpaw-tools** has evolved and is now called **Nanoworks**!

The **gpaw-tools** project began as a script that utilized only ASE and GPAW. Still, over the course of four years, it evolved into code that leverages multiple libraries, including ASAP3, Phonopy, Elastic, OpenKIM, and others. It is now being rewritten to incorporate modern Machine Learning capabilities (MACE, CHGNet, SevenNet) into its structure.

At this point, we have embarked on a new naming convention to better define the software. After this stage, it will be called `Nanoworks`. The `gpaw-tools` project has moved to `Nanoworks' new GitHub repository <https://github.com/sblisesivdin/nanoworks>`_ to achieve a more comprehensive, modern, and ML-supported structure.

What does this mean for you?

`gpaw-tools (v25.x)`: Has been placed in maintenance mode. No new features will be added except for critical bug fixes.

`Nanoworks (v26.x)`: Includes all the capabilities of gpaw-tools (excluding gg.py), but offers them as a modern Python package (pip install). It also incorporates modern machine learning capabilities such as MACE and CHGNet.

.. note::

   Nanoworks is in a beta phase. Please continue to use gpaw-tools until further notice. You can view the development at `Release Notes <https://github.com/sblisesivdin/nanoworks/blob/main/RELEASE_NOTES.md>`_


Nanoworks and gpaw-tools are distributed with `MIT license <https://github.com/sblisesivdin/nanoworks/blob/main/LICENSE.md>`_.

Check out the :doc:`usage` section for further information, including how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
--------

.. toctree::

   about
   usage
