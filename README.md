<div align="center">
  <img src="/images/logo.png" width="400" alt="Normal Splines">
</div>

# Multivariate Normal Hermite-Birkhoff Uniform Smoothing Splines in Julia

(project is under development)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://IgorKohan.github.io/NormalSmoothingSplines.jl/dev)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://IgorKohan.github.io/NormalSmoothingSplines.jl/stable)
[![Build Status](https://travis-ci.com/IgorKohan/NormalSmoothingSplines.jl.svg?branch=master)](https://travis-ci.com/github/IgorKohan/NormalSmoothingSplines.jl)
[![codecov](https://codecov.io/gh/IgorKohan/NormalSmoothingSplines.jl/branch/master/graph/badge.svg?token=ynsySRTggq)](https://codecov.io/gh/IgorKohan/NormalSmoothingSplines.jl)


![Problem definition](/images/problem-definition-smoothing.png)

Further examples are given in documentation.  

The normal splines method for one-dimensional function interpolation and linear ordinary differential and integral equations was proposed in [2]. An idea of the multivariate splines in Sobolev space was initially formulated in [8], however it was not well-suited to solving real-world problems. Using that idea the multivariate generalization of the normal splines method was developed for two-dimensional problem of low-range computerized tomography in [3] and applied for solving a mathematical economics problem in [4]. At the same time an interpolation scheme with Matérn kernels was developed in [9], this scheme coincides with interpolating normal splines method. Further results related to  applications of the normal splines method were reported at the seminars and conferences [5,6,7]. 

## Documentation

For more information and explanation see [Documentation](https://igorkohan.github.io/NormalSmoothingSplines.jl/stable/).

**References**

[1] [Halton sequence](https://en.wikipedia.org/wiki/Halton_sequence)

[2] V. Gorbunov, The method of normal spline collocation. [USSR Computational Mathematics and Mathematical Physics, Vol. 29, No. 1, 1989](https://www.researchgate.net/publication/265357408_Method_of_normal_spline-collocation)

[3] I. Kohanovsky, Normal Splines in Computing Tomography (Нормальные сплайны в вычислительной томографии). [Avtometriya, No.2, 1995](https://www.iae.nsk.su/images/stories/5_Autometria/5_Archives/1995/2/84-89.pdf) 

[4] V. Gorbunov, I. Kohanovsky, K. Makedonsky, Normal splines in reconstruction of multi-dimensional dependencies. [Papers of WSEAS International Conference on Applied Mathematics, Numerical Analysis Symposium, Corfu, 2004](http://www.wseas.us/e-library/conferences/corfu2004/papers/488-312.pdf)

[5] I. Kohanovsky, Multidimensional Normal Splines and Problem of Physical Field Approximation, International Conference on Fourier Analysis and its Applications, Kuwait, 1998.

[6] I. Kohanovsky, Inequality-Constrained Multivariate Normal Splines with Some Applications in Finance. [27th GAMM-Seminar on Approximation of Multiparametric functions](https://www.mis.mpg.de/scicomp/gamm27/Igor_Kohanovsky.pdf), Max-Planck-Institute for Mathematics in the Sciences, Leipzig, Germany, 2011.

[7] V. Gorbunov, I. Kohanovsky, Heterogeneous Parallel Method for the Construction of Multi-dimensional Smoothing Splines. [ESCO 2014 4th European Seminar on Computing, 2014](https://www.ana.iusiani.ulpgc.es/proyecto2015-2017/pdfnew/ESCO2014_Book_of_Abstracts.pdf)

[8] A. Imamov,  M. Dzhurabaev, Splines in S.L. Sobolev spaces (Сплайны в пространствах С.Л.Соболева). Deposited manuscript. Dep. UzNIINTI, No 880, 1989.

[9] J. Dix, R. Ogden, An Interpolation Scheme with Radial Basis in Sobolev Spaces H^s(R^n). [Rocky Mountain J. Math. Vol. 24, No.4,  1994.](https://projecteuclid.org/download/pdf_1/euclid.rmjm/1181072340)
