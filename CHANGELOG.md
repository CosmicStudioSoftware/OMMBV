# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [1.0.2] - ?????
- Switched away from distutils to Meson for build system
- Updated coupling to coveralls
- Updated package version for security issue in sphinx
- Added online unit testing for Linux/MacOS/Windows

## [1.0.1] - 2022-01-04
- Added pyproject.toml to support systems without numpy.
- Modified manifest.ini to include version.txt
- Corrected bad link in build status badge
- Corrected BSD license classifier

## [1.0.0] - 2021-12-24
- Updated vector basis algorithm and reduced uncertainty when 
scaling ion drifts and electric fields for multipole fields by four orders of magnitude.
- Added support for multiple unit test magnetic fields, from dipole up to octupole. 
- Added support for testing vector basis determination with a spherical Earth.
- Improved tolerance checks on vector basis during iteration so outputs better
reflect user settings.
- Improved robustness of vector basis calculation at higher latitudes.
- Updated IGRF step method to slow integration when reaching target altitude
  to use user provided step size.
- Allow automatic expansion of `field_line_trace` step_size after many iterations.
- Improved robustness of `apex_location_info` to unreported changes in 
`field_line_trace` step_size.
- Added `pole_tol` keyword which specifies how close to vertical local
  magnetic field must be to be considered a pole.
- Added `utils.datetimes_to_doubles` a method to calculate year and 
  fractional day of year for IGRF calls.
- Added keyword `max_steps` to `apex_location_info`
- Corrected normalization error in `magnetic_vector`.
- Deprecated `scalar` input for basis vectors.
- Reduced number of allowed recursive calls for `field_line_trace`
- Moved vector functions to `OMMBV.vector`
- Moved transformation functions to `OMMBV.trans`
- Moved tracing functions to `OMMBV.trace`
- Moved older algorithms not needed for current outputs to `OMMBV.heritage`
- Moved supporting Fortran functions from `OMMBV.igrf13` to `OMMBV.sources`
- Improved robustness of `apex_location_info` to `full_field_line` tracing failures.
- Added support for GitHub Workflows
- Refactored unit testing
- Moved to `setup.cfg`
- Added `.zenodo.json`
- Improved documentation
- Updated docstrings
- Added logo
- Updated testing versions

## [0.5.5] - 2021-06-16
- Updated setup.py
- Added compatibility with pysat v3.x

## [0.5.4] - 2020-03-03
- Updated to latest IGRF reference code

## [0.5.3] - 2020-06-19
- Metadata clarity improvements

## [0.5.2] - 2020-06-04
- Fixed bugs coupling the code repo to community services related to the name 
  change from pysatMagvect to OMMBV

## [0.5.0] - 2020-02-01
- Added community documents
- Implemented new algorithm for basis vectors. First system valid for multipole fields.
- Implemented new E and D scaling vectors similar to Richmond (apexpy)
- Implemented high accuracy numerical path that minimizes geodetic transformations
- Reviewed and updated default parameters based upon observed peformance
- Validated meridional vector along maximum apex height gradient
- Retained previous basis methods, identified by heritage or integrated
- Improved accuracy and robustness of heritage techniques though new methods recommended
- Validated accuracy of scaling methods
- Incorporated IGRF13
- Expanded and organized unit tests
- Removed pysat as a dependency
- Docstring improvements

## [0.4.0] - 2018-11-26
- Testing routines have been expanded significantly.
- Default parameters have been updated based upon this testing.
- Numerous corrections have been implemented.

## [0.3.1] - 2018-08-06
- Improved use of ECEF and ENU vectors.

## [0.3.0] - 2018-07-24
- Improved robustness of setup.py
- Corrected Earth Centered Earth Fixed (ECEF) and East, North, Up (ENU) conversion routines.

## [0.2.0] - 2018-07-05
- Improved documentation and metadata.

## [0.1.6] - 2018-06-29
- Improved installation process

## [0.1.0] - 2018-06-27
- Initial release
