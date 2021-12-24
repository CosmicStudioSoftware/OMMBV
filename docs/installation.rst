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
This module officially supports Python 3.6+.

 ============== =================
 Common modules Community modules
 ============== =================
  numpy         pysat
  scipy
 ============== =================


.. _install-opt:

Installation Options
--------------------

1. Clone the git repository
::


   git clone https://github.com/rstoneback/OMMBV.git


2. Install OMMBV:
   Change directories into the repository folder and run the setup.py file.
   There are a few ways you can do this:

   A. Install on the system (root privileges required)::


        sudo python3 setup.py install
   B. Install at the user level::


        python3 setup.py install --user
   C. Install with the intent to develop locally::


        python3 setup.py develop --user
