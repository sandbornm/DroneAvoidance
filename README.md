# Drone Collision Avoidance in MATLAB using Casadi and the PATH Solver

## Usage

This repo assumes Ubuntu >=18.04, MATLAB >=R2014b, and uses CasADI v3.5.5. The `pathm` directory contains the MATLAB source code and linux binary for the PATH solver. `test_path.py` contains a reference toa general optimization problem for drone dynamics whose solution is obtained by solving a [Nonlinear Mixed Complementarity Problem (NMCP)](https://en.wikipedia.org/wiki/Mixed_complementarity_problem).


Note: to solve problems of a certain size with the PATH solver, a [LICENSE](http://pages.cs.wisc.edu/~ferris/path/LICENSE) string is required. This should be set as an environment variable named `PATH_LICENSE_STRING` in a `.bashrc` file or the equivalent. To ensure the license string is set up correctly, run the following from this repository's path in MATLAB: `getenv("PATH_LICENSE_STRING")` should return the license string and `run runtests.m` should finish with `runtests completed OK`.


## References

- [PATH Solver](pages.cs.wisc.edu/~ferris/path.html)
- [Casadi](https://web.casadi.org)
