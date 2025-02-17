# Version history

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-xx-xx
### Changed
- Use h5netcdf instead of netcdf4 library

## [1.26.0] - 2025-01-23
### Added
- Special variable TIMESTEPS to be used as weighting parameter
- Special variable AREA to be used as weighting parameter
- New "aligned" parameter for binning

## [1.25.1] - 2025-01-22
### Added
- Geotag now accepts polygons with holes

## [1.24.7] - 2025-01-13
### Fixed
- Fix bug when weights parameter is pure number

## [1.24.6] - 2024-12-02
### Fixed
- Allow optional params in varfunc spec

## [1.24.5] - 2024-10-31
### Fixed
- Allow weights parameter to be pure number
- Bugfix in particle filter

## [1.24.4] - 2024-05-23
### Fixed
- Deprecation warnings

## [1.24.0] - 2024-01-17
### Added
- Option for supplying in-memory grid datasets to API functions

## [1.23.0] - 2024-01-16
### Added
- Option for supplying in-memory datasets to API functions
### Fixed
- Grid files no longer fail if they include CF time attributes

## [1.22.0] - 2023-12-12
### Removed
- Mandatory logging to file

## [1.21.0] - 2023-12-12
### Added
- API function for running simulations from another python script

## [1.20.0] - 2023-11-07
### Added
- New interpolation methods

## [1.19.3] - 2023-10-24
### Changed
- Uses github as CI instead of circleCI  

## [1.19.2] - 2023-08-28
### Fixed
- Accepts function expression with two dots (bug introduced in 1.18.2)  

## [1.19.1] - 2023-08-22
### Fixed
- Particle filter no longer breaks if pid count is large 

## [1.19] - 2023-08-18
### Added
- Derived variables

## [1.18.2] - 2023-08-17
### Fixed
- Weights expression can now contain floats

## [1.18.1] - 2023-06-29
### Fixed
- Log file is now closed after main script has ended

## [1.18] - 2023-06-29
### Added
- Allow arbritary time units in bin specification  
### Changed
- Reduced output verbosity of scan function
### Fixed
- Program can now be run as main module
- Weighted particles can now be outside range

## [1.17] - 2023-06-27
### Added
- Added logging to file
### Changed
- Reduced output verbosity
### Fixed
- No longer fails if there are spaces in time unit definition
- No longer fails if output varname is changed and projection is set

## [1.16] - 2023-06-27
### Added
- Option for changing output variable name

## [1.15] - 2023-06-26
### Added
- Alternative date specification method

## [1.14] - 2022-11-16
### Added
- Particle filter method

## [1.13] - 2022-10-26
### Added
- Fast timestep filter

## [1.12] - 2022-09-10
### Changed
- The main script name is now "crecon"

## [1.11] - 2022-08-15
### Fixed
- The variable name "weights" is no longer reserved

## [1.10] - 2022-07-08
### Added
- Allow projection information in multifile outputs
- Allow sampling of variables from grid files

## [1.9] - 2022-06-29
### Added
- Geotagging feature
- Connectivity computation
### Fixed
- Example "complex" now works from command line

## [1.8] - 2022-06-27
### Changed
- Optimized particle filtering

## [1.7] - 2022-06-24
### Added
- More logging statements
### Changed
- Optimized pre-scanning for particle variables
### Fixed
- Error in main script argument parsing 

## [1.6] - 2022-06-23
### Added
- Config script may now exclude certain explicit specifications, which are
  instead added by default.
- Complex example showing several features at once
- Projection information can now be added to output file
### Fixed
- Bug which caused adaptive histogram tests to fail on some python versions
- Bug which caused failure when all particles were outside range

## [1.5] - 2022-06-21
### Added
- Allow splitting by time steps

## [1.4] - 2022-06-20
### Added
- Group-by feature

## [1.3] - 2022-06-01
### Changed
- Histogram feature now uses pandas backend for improved scalability

## [1.2] - 2022-05-31
### Added
- Use multiple ladim input files
- Split output into multiple files
### Removed
- Autobin feature of Histogram class

## [1.1] - 2022-05-31
### Added
- Histogram weights based on numeric expression

## [1.0] - 2022-05-30
### Added
- Unlimited number of binning dimensions
- Filter particles based on numeric expression
