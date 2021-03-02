# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## [0.5.4] - 2020-03-03
- Updated to latest IGRF reference code

## [0.5.3] - 2020-06-19
- Metadata clarity improvements

## [0.5.2] - 2020-06-04
- Fixed bugs coupling the code repo to community services related to the name change from pysatMagvect to OMMBV

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
