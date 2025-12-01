# 📘 Numerical Analysis – Final Project  
**Ben-Gurion University of the Negev (BGU)**  
**Student: Dor Meir**

This repository contains my full implementation for the final project in the **Numerical Analysis** course.  
The project includes **five programming assignments**, each focusing on a core topic in numerical computation, interpolation, approximation, or integration.  

All methods were implemented **from scratch**, using only Python’s basic capabilities — without numerical libraries such as NumPy, SciPy, or SymPy — in order to demonstrate a deep understanding of numerical algorithms and their mathematical foundations.

---

# 🚀 Project Overview

The project is divided into five independent tasks:

1. **Function interpolation using Bézier splines**  
2. **Finding multiple intersection points between two functions**  
3. **Gaussian Quadrature integration & area between curves**  
4. **Denoising a function and curve fitting**  
5. **Fitting a noisy 2D shape and computing its area**

Each task has its own file in the repository, with clean separation of logic and helper functions.

The final submitted report is also included as `numerical_analysis_final_report.pdf`.

---

# 📁 Repository Structure

```
├── function_interpolation_bezier_splines.py
├── multiple_intersections_regula_falsi.py
├── gaussian_quadrature_integration_area_between_curves.py
├── noisy_curve_fitting_bezier_denoising.py
├── noisy_shape_fitting_and_polygon_area.py
├── numerical_analysis_final_report.pdf
└── README.md
```

Below is a full explanation of what each file contains.

---

# 🧩 **1. Function Interpolation Using Bézier Splines**  
_File: `function_interpolation_bezier_splines.py`_

This module implements a smooth interpolation of a given continuous function over an interval [a, b] using **cubic Bézier segments**.

### ✔ Key Components
- Uniform sampling of the function at N+1 points  
- Construction of piecewise cubic Bézier curves  
- Solving a **tridiagonal linear system** (Thomas algorithm) to compute the control points  
- Returning a smooth interpolant that can be evaluated anywhere on [a, b]

### ✔ Core Numerical Topics
- Bézier representation  
- Smooth curve stitching  
- Tridiagonal system solving  
- Hermite-style constraints

---

# 🧮 **2. Finding Multiple Intersections With Regula Falsi**  
_File: `multiple_intersections_regula_falsi.py`_

This module finds **all intersection points** between two continuous functions f₁(x) and f₂(x) over a given interval.

### ✔ Key Components
- Scanning the interval and identifying potential sign-change regions  
- Applying the **Regula Falsi (False Position)** method to approximate each root  
- Handling:
  - flat intersections  
  - repeated intersections  
  - deduplication of close roots  
  - tolerance-based filtering

### ✔ Core Numerical Topics
- Root finding  
- Sign-change detection  
- Stability around near-flat intersections

---

# 📐 **3. Gaussian Quadrature & Area Between Curves**  
_File: `gaussian_quadrature_integration_area_between_curves.py`_

This file implements:

### **A. Numeric integration using Gaussian Quadrature**
Under a **limited function-calls budget**, the integrator chooses among:
- 1-point,
- 2-point,
- 3-point,
- 6-point,
- 10-point Gaussian quadrature.

It dynamically selects step sizes and quadrature order to stay within the evaluation budget while maximizing accuracy.

### **B. Area between two curves**
To compute the area enclosed by f₁(x) and f₂(x):

1. Intersection points are computed using the method from Assignment 2  
2. The interval is split into monotonic sub-segments  
3. ∫ |f₁(x) – f₂(x)| dx is evaluated using Gaussian quadrature

### ✔ Core Numerical Topics
- Approximation of definite integrals  
- Gaussian Quadrature theory  
- Error reduction via adaptive partitioning  
- Composite integration  
- Handling non-simple shapes

---

# 🎯 **4. Denoising & Curve Fitting Using Bézier Splines**  
_File: `noisy_curve_fitting_bezier_denoising.py`_

This algorithm receives **noisy function values** sampled at random points and fits a smooth approximating curve.

### ✔ Methodology
1. Sample the noisy function many times  
2. Average the values to reduce noise  
3. Fit a smooth curve using the **same Bézier-spline method** from Assignment 1  
4. Return a callable function representing the denoised curve

### ✔ Core Numerical Topics
- Noise reduction by resampling  
- Curve smoothing  
- Bézier spline reconstruction  
- Stable interpolation under noise

---

# 🔷 **5. Noisy Shape Fitting & Polygon Area**  
_File: `noisy_shape_fitting_and_polygon_area.py`_

This task deals with noisy samples from a **closed 2D shape**.  
The goal is to reconstruct the shape and compute its area.

### ✔ Key Components
- Sampling noisy points along the contour  
- Recentering and sorting points by polar angle  
- Grouping points into segments (clustering)  
- Averaging points per segment to reduce noise  
- Fitting a cleaned contour  
- Computing area with the **Shoelace Formula**

### ✔ Core Numerical Topics
- Geometric denoising  
- Curve reconstruction from unordered points  
- Polygon area estimation  
- Robust clustering and ordering

---

# 📄 Final Report

The file `numerical_analysis_final_report.pdf` contains:

- Mathematical derivations  
- Algorithm explanations  
- Example results  
- Full answers submitted as part of the final assignment

---

# 🛠 Requirements & Usage

The implementations rely only on:

- Standard Python (no NumPy, SciPy, etc.)
- Basic math and control structures

To use any of the modules:

```python
from function_interpolation_bezier_splines import interpolate
from gaussian_quadrature_integration_area_between_curves import integrate
```

Each file is fully self-contained.

---

# 🙌 Author

**Dor Meir**  
Ben-Gurion University of the Negev (BGU)  
M.Sc. in Information Systems Engineering  
B.Sc. in Data Science Engineering
