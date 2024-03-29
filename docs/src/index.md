```@meta
Author = "Igor Kohanovsky"
```

# NormalSmoothingSplines.jl package

*Multivariate Normal Hermite-Birkhoff Uniform Smoothing Splines in Julia*

`NormalSmoothingSplines.jl` implements the normal splines method for solving the following approximation problem:


*Problem:* Given points ``\{p_i, p_i \in R^n\}_{i=1}^{n_1}``, ``\{s_j, s_j \in R^n\}_{j=1}^{n_2}``, and ``\{\overline p_r, \overline p_r \in R^n\}_{r=1}^{n_3}``, ``\{\overline s_t, \overline s_t \in R^n\}_{t=1}^{n_4}`` and sets of unit vectors ``\{e_j, e_j \in R^n\}_{r=1}^{n_2}``, ``\{\overline e_t, \overline e_t \in R^n\}_{t=1}^{n_4}`` find a function ``f`` such that

```math
\tag{1}
\begin{aligned}
& f(p_i) =  u_i \, , \quad  i = 1, 2, \dots, n_1 \, ,
\\  
& \frac{ \partial{f} }{ \partial{e_j} }(s_j) =  v_j \, , \quad  j = 1, 2, \dots, n_2 \, ,
\\  
&  \underline u_r \le f(\overline p_r) \le \overline u_r \, , \quad  r = 1, 2, \dots, n_3 \, ,
\\  
&  \underline v_t \le \frac{ \partial{f} }{ \partial{\overline e_t} } (\overline s_t) \le \overline v_t \, , \quad  t = 1, 2, \dots, n_4 \, , \\
& n_1 \ge 0 \, , \  n_2 \ge 0 \, , \ n_3 \ge 0 \, , \  n_4 \ge 0 \, ,
\end{aligned}
``` 
where ``\frac{ \partial{f} }{ \partial{e} }(s) = \nabla f(s) \cdot e = \sum _{k=1}^{n}  \frac{ \partial{f} }{ \partial{x_k} } (s) e_{k}`` is a directional derivative of function ``f`` at the point ``s`` in the direction of ``e``,
and  points ``\{p_i\}_{i=1}^{n_1}``, ``\{\overline p_r\}_{r=1}^{n_3}`` as well as points ``\{s_j\}_{j=1}^{n_2}``, ``\{\overline s_t\}_{t=1}^{n_4}`` are pairwise different.

We assume that function ``f`` is an element of the Bessel potential space ``H^s_\varepsilon (R^n)`` which is defined as:

```math
   H^s_\varepsilon (R^n) = \left\{ \varphi | \varphi \in S' ,
  ( \varepsilon ^2 + | \xi |^2 )^{s/2}{\mathcal F} [\varphi ] \in L_2 (R^n) \right\} , \quad
  \varepsilon \gt 0 , \  s = n/2 + 1/2 + r \, , \quad r = 1,2,\dots \, .
```
where ``| \cdot |`` is the Euclidean norm, ``S'  (R^n)`` is space of L. Schwartz tempered distributions, parameter ``s`` may be treated as a fractional differentiation order and ``\mathcal F [\varphi ]`` is a Fourier transform of the ``\varphi``. The parameter ``\varepsilon`` can be considered as a "scaling parameter", it allows to control approximation properties of the normal spline which usually are getting better with smaller values of ``\varepsilon``, also it can be used to reduce the ill-conditioness of the related computational problem (in traditional theory ``\varepsilon = 1``).

The Bessel potential space ``H^s_\varepsilon (R^n)`` is a  Reproducing kernel Hilbert space, an element ``f`` of that space can be treated as a ``r``-times continuously differentiable function.

The normal splines method consists in finding a solution of system (1) having minimal norm in Hilbert space ``H^s_\varepsilon (R^n)``, thus an uniform smoothing normal spline ``\sigma`` is defined as follows:

```math
\tag{2}
   \sigma = {\rm arg\,min}\{  \| f \|^2 : (1), \ \forall f \in H^s_\varepsilon (R^n) \} \, .
```

The normal splines method is based on the following functional analysis results:

* Bessel potential space embedding theorem
* The Riesz representation theorem for Hilbert spaces
* Reproducing kernel properties 

Using these results it is possible to reduce task (2) to solving a finite-dimensional quadratic programming problem. 

The normal splines method for one-dimensional function interpolation and linear ordinary differential and integral equations was proposed in [1]. An idea of the multivariate splines in Sobolev space was initially formulated in [7], however it was not well-suited to solving real-world problems. Using that idea the multivariate generalization of the normal splines method was developed for two-dimensional problem of low-range computerized tomography in [2] and applied for solving a mathematical economics problem in [3]. At the same time an interpolation scheme with Matérn kernels was developed in [8], this scheme coincides with interpolating normal splines method. Further results related to  applications of the normal splines method were reported at the seminars and conferences [4,5,6]. 

#### References:

[1] V. Gorbunov, The method of normal spline collocation. [USSR Comput.Maths.Math.Phys., Vol. 29, No. 1, 1989](https://www.sciencedirect.com/science/article/abs/pii/0041555389900591)

[2] I. Kohanovsky, Normal Splines in Computing Tomography (Нормальные сплайны в вычислительной томографии). [Avtometriya, No.2, 1995](https://www.iae.nsk.su/images/stories/5_Autometria/5_Archives/1995/2/84-89.pdf)

[3] V. Gorbunov, I. Kohanovsky, K. Makedonsky, Normal splines in reconstruction of multi-dimensional dependencies. [Papers of WSEAS International Conference on Applied Mathematics, Numerical Analysis Symposium, Corfu, 2004](http://www.wseas.us/e-library/conferences/corfu2004/papers/488-312.pdf)

[4] I. Kohanovsky, Multidimensional Normal Splines and Problem of Physical Field Approximation, International Conference on Fourier Analysis and its Applications, Kuwait, 1998.

[5] I. Kohanovsky, Inequality-Constrained Multivariate Normal Splines with Some Applications in Finance. [27th GAMM-Seminar on Approximation of Multiparametric functions](https://www.mis.mpg.de/scicomp/gamm27/Igor_Kohanovsky.pdf), Max-Planck-Institute for Mathematics in the Sciences, Leipzig, Germany, 2011.

[6] V. Gorbunov, I. Kohanovsky, Heterogeneous Parallel Method for the Construction of Multi-dimensional Smoothing Splines. [ESCO 2014 4th European Seminar on Computing](https://www.ana.iusiani.ulpgc.es/proyecto2015-2017/pdfnew/ESCO2014_Book_of_Abstracts.pdf), University of West Bohemia, Plzen, Czech Republic, 2014.

[7] A. Imamov,  M. Dzhurabaev, Splines in S.L. Sobolev spaces (Сплайны в пространствах С.Л.Соболева). Deposited manuscript. Dep. UzNIINTI, No 880, 1989.

[8] J. Dix, R. Ogden, An Interpolation Scheme with Radial Basis in Sobolev Spaces H^s(R^n). [Rocky Mountain J. Math. Vol. 24, No.4,  1994.](https://projecteuclid.org/download/pdf_1/euclid.rmjm/1181072340)

## Contents

```@contents
Pages = [
      "index.md",
      "Public-API.md",
      "Usage.md",
      "Normal-Splines-Method.md"
]
Depth = 3
```
