# Course Design – 1D Poisson Equation (Dirichlet BC)

This repository aims to solve the one-dimensional Poisson equation with Dirichlet boundaries by iterative methods such as **Jacobi**, **Gauss–Seidel (G-S)** and **SOR**.

## Files

- `1d.py`  
  A simple test program to solve the 1D discrete Poisson equation with Jacobi / G-S / SOR and plot the solution and residual convergence. (Not used directly in the report.)

- `1ddilichlet.py`  
  Main program to solve the 1D discrete Poisson equation with Dirichlet boundaries using Jacobi, G-S and SOR, and compare their iteration counts, errors and runtime.

- `1dSOR.py`  
  Example script using SOR with a fixed relaxation factor to show the numerical solution and convergence behavior.

- `1dSORomega.py`  
  Example script to scan different SOR relaxation factors \(\omega\) and study how \(\omega\) affects convergence.

- `1dSORstepsize.py`  
  Example script to study how the grid size (step size) influences SOR convergence and accuracy.

- `求解离散泊松方程的三种迭代方法比较.pdf`  
  Short report (in Chinese) summarizing the model, the three iterative methods, and numerical comparison results.

