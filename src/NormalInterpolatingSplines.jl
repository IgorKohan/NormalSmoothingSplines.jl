#### Interpolating splines inteface deinition
export prepare, construct, interpolate
export evaluate, evaluate_one, evaluate_gradient
# -- 1D case --
export evaluate_derivative
# --

"""
`prepare(nodes::Matrix{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare the interpolating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
function prepare(nodes::Matrix{T},
                 kernel::RK = RK_H0()
                ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare(nodes, kernel)
     return spline
end

"""
`construct(spline::NormalSpline{T, RK}, values::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

construct the interpolating normal spline by calculating its coefficients and
completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at interpolation nodes.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
function construct(spline::NormalSpline{T, RK},
                   values::Vector{T}
                  ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    spline = _construct(spline, values)
    return spline
end

"""
`interpolate(nodes::Matrix{T}, values::Vector{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare and construct the interpolating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `values`: function values at interpolation nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
function interpolate(nodes::Matrix{T},
                     values::Vector{T},
                     kernel::RK = RK_H0()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare(nodes, kernel)
     spline = _construct(spline, values)
     return spline
end

"""
`evaluate(spline::NormalSpline{T, RK}, points::Matrix{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the interpolating spline values at the locations defined in `points`.

# Arguments
- `spline: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `points`: locations at which spline values are evaluating.
            This should be an `n×m` matrix, where `n` is dimension of the sampled space
            and `m` is the number of locations where spline values are evaluating.
            It means that each column in the matrix defines one location.

Return: `Vector{T}` of the spline values at the locations defined in `points`.
"""
function evaluate(spline::NormalSpline{T, RK},
                  points::Matrix{T}
                 ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate(spline, points)
end

"""
`evaluate_one(spline::NormalSpline{T, RK}, point::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the interpolating spline value at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Return: the spline value at the location defined in `point`.
"""
function evaluate_one(spline::NormalSpline{T, RK},
                      point::Vector{T}
                     ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate(spline, reshape(point, :, 1))[1]
end

"""
`evaluate_gradient(spline::NormalSpline{T, RK}, point::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate gradient of the spline at the location defined in `point`.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which gradient value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Note: Gradient of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

Return: `Vector{T}` - gradient of the spline at the location defined in `point`.
"""
function evaluate_gradient(spline::NormalSpline{T, RK},
                           point::Vector{T}
                          ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate_gradient(spline, point)
end

########

"""
`prepare(nodes::Matrix{T}, d_nodes::Matrix{T}, es::Matrix{T}, kernel::RK = RK_H1())
         where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Prepare the spline by constructing and factoring a Gram matrix of the interpolation problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes.
            It means that each column in the matrix defines one node.
- `d_nodes`: function directional derivatives nodes.
             This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
             `n_2` is the number of function directional derivative nodes.
- `es`: Directions of the function directional derivatives.
        This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
        `n_2` is the number of function directional derivative nodes.
        It means that each column in the matrix defines one direction of the function directional derivative.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
function prepare(nodes::Matrix{T},
                 d_nodes::Matrix{T},
                 es::Matrix{T},
                 kernel::RK = RK_H1()
                ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     spline = _prepare(nodes, d_nodes, es, kernel)
     return spline
end

"""
`construct(spline::NormalSpline{T, RK}, values::Vector{T}, d_values::Vector{T})
           where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Construct the interpolating normal spline by calculating its coefficients and
completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at interpolation nodes.
- `d_values`: function directional derivative values at function derivative nodes.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
function construct(spline::NormalSpline{T, RK},
                   values::Vector{T},
                   d_values::Vector{T}
                  ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     spline = _construct(spline, values, d_values)
     return spline
end

"""
`interpolate(nodes::Matrix{T}, values::Vector{T}, d_nodes::Matrix{T}, es::Matrix{T}, d_values::Vector{T}, kernel::RK = RK_H1())
             where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Prepare and construct the interpolating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `values`: function values at interpolation nodes.
- `d_nodes`: function directional derivative nodes.
            This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
            `n_2` is the number of function directional derivative nodes.
- `es`: Directions of the function directional derivatives.
       This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
       `n_2` is the number of function directional derivative nodes.
       It means that each column in the matrix defines one direction of the function directional derivative.
- `d_values`: function directional derivative values at function derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
function interpolate(nodes::Matrix{T},
                     values::Vector{T},
                     d_nodes::Matrix{T},
                     es::Matrix{T},
                     d_values::Vector{T},
                     kernel::RK = RK_H1()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     spline = _prepare(nodes, d_nodes, es, kernel)
     spline = _construct(spline, values, d_values)
     return spline
end

############################## One-dimensional case

"""
`prepare(nodes::Vector{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare the 1D interpolating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
function prepare(nodes::Vector{T},
                 kernel::RK = RK_H0()
                ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare(Matrix(nodes'), kernel)
     return spline
end

"""
`interpolate(nodes::Vector{T}, values::Vector{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare and construct the 1D interpolating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `values`: function values at `n_1` interpolation nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
function interpolate(nodes::Vector{T},
                     values::Vector{T},
                     kernel::RK = RK_H0()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare(Matrix(nodes'), kernel)
     spline = _construct(spline, values)
     return spline
end

"""
`evaluate(spline::NormalSpline{T, RK}, points::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the 1D interpolating spline values/value at the `points` locations.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `points`: locations at which spline values are evaluating.
            This should be a vector of size `m` where `m` is the number of evaluating points.

Return: spline value at the `point` location.
"""
function evaluate(spline::NormalSpline{T, RK},
                  points::Vector{T}
                 ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate(spline, Matrix(points'))
end

"""
`evaluate_one(spline::NormalSpline{T, RK}, point::T) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the 1D interpolating spline value at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline value is evaluating.

Return: spline value at the `point` location.
"""
function evaluate_one(spline::NormalSpline{T, RK},
                      point::T
                     ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    v_points = Vector{T}(undef, 1)
    v_points[1] = point
    return _evaluate(spline, Matrix(v_points'))[1]
end

"""
`evaluate_derivative(spline::NormalSpline{T, RK}, point::T) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the 1D spline derivative at the `point` location.

# Arguments
- `spline`: the `NormalSpline` object returned by `interpolate` or `construct` function.
- `point`: location at which spline derivative is evaluating.

Note: Derivative of spline built with reproducing kernel RK_H0 does not exist at the spline nodes.

Return: the spline derivative value at the `point` location.
"""
function evaluate_derivative(spline::NormalSpline{T, RK},
                             point::T
                            ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    v_points = Vector{T}(undef, 1)
    v_points[1] = point
    return _evaluate_gradient(spline, v_points)[1]
end

"""
`prepare(nodes::Vector{T}, d_nodes::Vector{T}, kernel::RK = RK_H1()) where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Prepare the 1D interpolating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `d_nodes`: The function derivatives nodes.
             This should be an `n_2` vector where `n_2` is the number of function derivatives nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the partly initialized `NormalSpline` object that must be passed to `construct` function
        in order to complete the spline initialization.
"""
function prepare(nodes::Vector{T},
                 d_nodes::Vector{T},
                 kernel::RK = RK_H1()
                ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}

     es = ones(T, length(d_nodes))
     spline = _prepare(Matrix(nodes'), Matrix(d_nodes'), Matrix(es'), kernel)
     return spline
end

"""
`interpolate(nodes::Vector{T}, values::Vector{T}, d_nodes::Vector{T}, d_values::Vector{T},
             kernel::RK = RK_H1())
             where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Prepare and construct the 1D interpolating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value nodes.
- `values`: function values at `nodes` nodes.
- `d_nodes`: The function derivatives nodes.
             This should be an `n_2` vector where `n_2` is the number of function derivatives nodes.
- `d_values`: function derivative values at function derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: the completely initialized `NormalSpline` object that can be passed to `evaluate` function.
"""
function interpolate(nodes::Vector{T},
                     values::Vector{T},
                     d_nodes::Vector{T},
                     d_values::Vector{T},
                     kernel::RK = RK_H1()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     es = ones(T, length(d_nodes))
     spline = _prepare(Matrix(nodes'), Matrix(d_nodes'), Matrix(es'), kernel)
     spline = _construct(spline, values, d_values)
     return spline
end
