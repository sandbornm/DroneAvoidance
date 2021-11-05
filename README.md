# Drone Collision Avoidance in MATLAB using Casadi and the PATH Solver

## Usage

This repo assumes Ubuntu >=18.04, MATLAB >=R2014b, and uses CasADI v3.5.5. The `pathm` directory contains the source code and linux binary for the PATH solver. `test_path.py` contains a reference toa general optimization problem for drone dynamics whose solution is obtained by solving a [Nonlinear Mixed Complementarity Problem (NMCP)](https://en.wikipedia.org/wiki/Mixed_complementarity_problem).

## References

- [PATH Solver](pages.cs.wisc.edu/~ferris/path.html)
- [Casadi](https://web.casadi.org)
