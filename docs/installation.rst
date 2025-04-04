.. _install:

Installation
============

The following instructions will allow you to install OMMBV.


.. _install-prereq:

Prerequisites
-------------

.. image:: images/poweredbypysat.png
    :width: 150px
    :align: right
    :alt: powered by pysat Logo, blue planet with orbiting python


OMMBV uses common Python modules, as well as modules developed by and for
the Space Physics community. pysat is an optional module, used by OMMBV
to make it easy for satellite missions to add OMMBV results.
This module officially supports Python 3.9+.

 ============== =================
 Common modules Community modules
 ============== =================
  numpy         pysat (optional)
  scipy
 ============== =================


.. _install-opt:

Installation Options
--------------------

A. Using pypi ::

    pip install OMMBV


B. OMMBV may also be installed directly from the source repository on github:

   1. Clone the git repository::

         git clone https://github.com/CosmicStudioSoftware/OMMBV.git



   2. Install OMMBV: ::

        cd OMMBV
        pip install --user .

