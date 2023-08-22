# Version history

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
