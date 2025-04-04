.. _overview:

Overview
========

The derivation of an appropriate vector basis has been a long-term challenge
in space science. This package provides the first accurate orthogonal
vector basis that supports mapping electric fields or other parameters at one
location to any other location on the same field line.
OMMBV retains these properties for non-spherical planets, such as the Earth, as
well as for magnetic fields that are more complicated than a pure dipole, such
as the Earth's.

These features are fundamental for Space Science studies. Satellite measurements
of plasma motions in-situ can now be accurately mapped along field lines
for comparison to ground based equipment. Forcing from the neutral atmosphere
may now also be mapped to satellite altitudes to accurately characterize
both the magnitude and direction of that forcing on plasma motion at satellite
locations. Computer modelers may use OMMBV to map electric fields calculated
within a single plane through the whole magnetosphere, significantly reducing
computational resources.

OMMBV has been validated using a variety of test magnetic sources as well as
via application to the Earth's magnetic field using the
`International Geomagnetic Reference Field (IGRF) <https://geomag.bgs.ac.uk/research/modelling/IGRF.html>`_.

.. image:: images/logo_high_res.png
    :width: 400px
    :align: center
    :alt: OMMBV Logo, Derived apex altitude with module name superimposed.
