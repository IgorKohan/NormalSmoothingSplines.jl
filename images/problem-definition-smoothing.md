This Julia package implements the normal splines method for solving the following approximation problem:

*Problem:* Given pairwise different points ``\{p_i, p_i \in R^n\}_{i=1}^{n_1}``, ``\{\overline p_i, \overline p_i \in R^n\}_{i=1}^{n_2}``, pairwise different points ``\{s_j, s_j \in R^n\}_{j=1}^{n_3}``, ``\{\overline s_j, \overline s_j \in R^n\}_{j=1}^{n_4}`` and sets of unit vectors ``\{e_j, e_j \in R^n\}_{j=1}^{n_3}``, ``\{\overline e_j, \overline e_j \in R^n\}_{j=1}^{n_4}`` find a function ``f`` such that

```math
\tag{1}
\begin{aligned}
& f(p_i) =  u_i \, , \quad  i = 1, 2, \dots, n_1 \, ,
\\  
&  \underline u_i \le f(\overline p_i) \le \overline u_i \, , \quad  i = 1, 2, \dots, n_2 \, ,
\\  
& \frac{ \partial{f} }{ \partial{e_j} }(s_j) =  v_j \, , \quad  j = 1, 2, \dots, n_3 \, ,
\\  
&  \underline v_j \le \frac{ \partial{f} }{ \partial{\overline e_j} } (\overline s_j) \le \overline v_j \, , \quad  j = 1, 2, \dots, n_4 \, , \\
& n_1 \ge 0 \, , \  n_2 \ge 0 \, , \ n_3 \ge 0 \, , \  n_4 \ge 0 \, ,
\end{aligned}
``` 
where ``\frac{ \partial{f} }{ \partial{e} }(s) = \nabla f(s) \cdot e = \sum _{k=1}^{n}  \frac{ \partial{f} }{ \partial{x_k} } (s) e_{k}`` is a directional derivative of function ``f`` at the point ``s`` in the direction of ``e``.

We assume that function ``f`` is an element of the Bessel potential space ``H^s_\varepsilon (R^n)`` which is defined as:

```math
   H^s_\varepsilon (R^n) = \left\{ \varphi | \varphi \in S' ,
  ( \varepsilon ^2 + | \xi |^2 )^{s/2}{\mathcal F} [\varphi ] \in L_2 (R^n) \right\} , \quad
  \varepsilon \gt 0 , \ \ s = n/2 + 1/2 + r \, , \quad r = 1,2,\dots \, .
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

Using these results it is possible to reduce task (2) to solving a finite-dimensional quadratic programming problem. Detailed explanation is given in the package documentation.
