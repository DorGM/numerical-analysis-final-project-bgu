# Numerical Analysis Algorithms in Python

A collection of numerical-analysis algorithms I implemented from scratch in Python, covering interpolation, root finding, Gaussian quadrature, noisy-curve fitting, and 2D shape reconstruction.

The core numerical methods are implemented directly rather than delegated to high-level numerical solvers. NumPy is used primarily for arrays, numerical primitives, and vectorized arithmetic.

The five modules can be imported independently.

## Overview

This project implements interpolation, root finding, Gaussian quadrature, noisy-curve fitting, and noisy 2D shape reconstruction in Python.

I implemented the core numerical algorithms from scratch. The methods live directly in this repository rather than being delegated to high-level numerical solvers. NumPy is used for low-level numerical primitives rather than as a replacement for the algorithms.

## Key Technical Highlights

- Cubic Bézier interpolation with a manually implemented Thomas / TDMA tridiagonal solver
- Multiple-intersection detection using interval scanning and Regula Falsi
- Gaussian quadrature with explicitly coded nodes and weights, under a function-evaluation budget
- Repeated sampling and averaging to denoise a callable, then Bézier-based curve fitting
- Polar-angle contour reconstruction from noisy 2D samples, with shoelace area
- Evaluation-budget arguments such as `n` in the public APIs

## Algorithms Implemented

### 1. Cubic Bézier interpolation — `Assignment1.interpolate`

Samples a callable on `[a, b]` with at most `n` evaluations, builds piecewise cubic Bézier segments, and solves a tridiagonal system (Thomas / TDMA) for the control points. Evaluating the interpolant uses a Regula Falsi helper to recover the Bézier parameter `t` for a given `x`.

### 2. Multiple intersections — `Assignment2.intersections`

Scans `[a, b]` for sign changes (and nearby near-flat candidates), then applies Regula Falsi on each bracket. Nearby candidate roots are filtered and deduplicated before the method returns an iterable of approximate intersection abscissae.

### 3. Gaussian quadrature and area between curves — `Assignment3`

`integrate(f, a, b, n)` composes 1-, 2-, 3-, 6-, and 10-point Gaussian rules whose nodes and weights are written out in the source. The composition is chosen so that `f` is not evaluated more than `n` times.

`areabetween(f1, f2)` locates intersections on the hard-coded interval `[1, 100]`, then integrates `|f1 - f2|` between consecutive roots. If fewer than two intersections are found, the implementation returns `0.0` as `float32` rather than NaN.

### 4. Noisy curve fitting — `Assignment4A.fit`

Repeatedly samples a noisy callable, averages the samples, and fits a cubic Bézier interpolant with the same TDMA control-point construction used in Assignment 1.

The method signature includes `d` (expected polynomial degree) and `maxtime`. Those parameters are accepted for API compatibility; the live fitting path does not currently use them.

### 5. Noisy shape reconstruction — `Assignment5`

`fit_shape` draws a large number of noisy contour samples, recenters them, sorts by polar angle, bins and averages points, and returns a shape object. Area is computed with the shoelace formula via NumPy `dot` / `roll`.

`area(contour)` samples a contour callable and applies the same shoelace formula.

The `maxtime` argument is part of the method signature and is not used by the live reconstruction path.

## From-Scratch Numerical Implementations

I implemented the core numerical algorithms from scratch. The following methods are implemented directly in this repository:

- Cubic Bézier construction and evaluation
- Thomas algorithm (TDMA) for tridiagonal control-point systems
- Regula Falsi, including a scan for multiple roots
- Gaussian quadrature rules (1, 2, 3, 6, and 10 points) with explicit nodes and weights
- Composite integration under an evaluation budget
- Repeated-sample denoising and Bézier curve fitting
- Polar-angle ordering, binning, and averaging for noisy closed contours
- Shoelace polygon area

NumPy and the Python standard library are used for:

- arrays and numerical dtypes (`float32`)
- `linspace` / `arange`
- vectorized arithmetic
- `dot` / `roll`
- scalar math (`math.sqrt`, `math.hypot`, `math.atan2`, …)

They are not used as replacements for SciPy-style interpolation, root-finding, integration, or linear-system solvers. NumPy is used for low-level numerical primitives rather than as a replacement for the algorithms.

The original project’s embedded test/demo code references `sampleFunctions` and `functionUtils`, which are not included in this repository. They are relevant only to that old test/demo code and are not required to import or call the core algorithm modules.

## Dependencies & Setup

Runtime dependency:

```text
numpy
```

```bash
pip install -r requirements.txt
```

Optional historical test/demo dependencies (`tqdm`, `matplotlib`, and the helper modules referenced above) are not required to use the algorithms.

## Usage

```python
from algorithms.interpolation import Assignment1

ass1 = Assignment1()
f = ass1.interpolate(lambda x: x, 0, 1, 5)
print(f(0.5))
```

```python
from algorithms.gaussian_quadrature import Assignment3

ass3 = Assignment3()
print(ass3.integrate(lambda x: x**2, 0, 1, 10))
```

```python
from algorithms.root_finding import Assignment2
from algorithms.curve_fitting import Assignment4A
from algorithms.shape_reconstruction import Assignment5

ass2 = Assignment2()
print(ass2.intersections(lambda x: x * x - 1, lambda x: 0, -2, 2))

ass4 = Assignment4A()
fit = ass4.fit(lambda x: x**2, 0, 1, d=2, maxtime=1.0)
print(fit(0.5))

ass5 = Assignment5()
```

## Project Structure

```text
README.md
requirements.txt
.gitignore
algorithms/
    __init__.py
    interpolation.py
    root_finding.py
    gaussian_quadrature.py
    curve_fitting.py
    shape_reconstruction.py
```

## Background

This project was developed while studying Numerical Analysis at Ben-Gurion University of the Negev (BGU). The five modules originated from numerical-analysis programming tasks covering interpolation, root finding, quadrature, curve fitting, and shape reconstruction.

## Author

**Dor Meir**  
Ben-Gurion University of the Negev (BGU)  
M.Sc. in Information Systems Engineering  
B.Sc. in Data Science Engineering
