"""
`prepare_interpolation(nodes::Matrix{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

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

Return: the prepared `NormalSpline` object.
"""
function prepare_interpolation(nodes::Matrix{T},
                               kernel::RK = RK_H0()
                              ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_interpolation(nodes, kernel)
     return spline
end

"""
`construct_interpolation(spline::NormalSpline{T, RK}, values::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Construct the interpolating normal spline by calculating its coefficients and
completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the prepared `NormalSpline` object.
- `values`: function values at interpolation nodes.

Return: the constructed `NormalSpline` object.
"""
function construct_interpolation(spline::NormalSpline{T, RK},
                                 values::Vector{T}
                                ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    spline = _construct_interpolation(spline, values)
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

Return: the constructed `NormalSpline` object.
"""
function interpolate(nodes::Matrix{T},
                     values::Vector{T},
                     kernel::RK = RK_H0()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_interpolation(nodes, kernel)
     spline = _construct_interpolation(spline, values)
     return spline
end

########

"""
`prepare_interpolation(nodes::Matrix{T}, d_nodes::Matrix{T}, es::Matrix{T}, kernel::RK = RK_H1())
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

Return: the prepared `NormalSpline` object.
"""
function prepare_interpolation(nodes::Matrix{T},
                               d_nodes::Matrix{T},
                               es::Matrix{T},
                               kernel::RK = RK_H1()
                              ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     spline = _prepare_interpolation(nodes, d_nodes, es, kernel)
     return spline
end

"""
`construct_interpolation(spline::NormalSpline{T, RK}, values::Vector{T}, d_values::Vector{T})
                         where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Construct the interpolating normal spline by calculating its coefficients and
completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the prepared `NormalSpline` object.
- `values`: function values at interpolation nodes.
- `d_values`: function directional derivative values at function derivative nodes.

Return: the constructed `NormalSpline` object.
"""
function construct_interpolation(spline::NormalSpline{T, RK},
                                 values::Vector{T},
                                 d_values::Vector{T}
                                ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     spline = _construct_interpolation(spline, values, d_values)
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

Return: the constructed `NormalSpline` object.
"""
function interpolate(nodes::Matrix{T},
                     values::Vector{T},
                     d_nodes::Matrix{T},
                     es::Matrix{T},
                     d_values::Vector{T},
                     kernel::RK = RK_H1()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     spline = _prepare_interpolation(nodes, d_nodes, es, kernel)
     spline = _construct_interpolation(spline, values, d_values)
     return spline
end

############################## One-dimensional case

"""
`prepare_interpolation(nodes::Vector{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

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

Return: the prepared `NormalSpline` object.
"""
function prepare_interpolation(nodes::Vector{T},
                               kernel::RK = RK_H0()
                              ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_interpolation(Matrix(nodes'), kernel)
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

Return: the pconstructed `NormalSpline` object.
"""
function interpolate(nodes::Vector{T},
                     values::Vector{T},
                     kernel::RK = RK_H0()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_interpolation(Matrix(nodes'), kernel)
     spline = _construct_interpolation(spline, values)
     return spline
end

"""
`prepare_interpolation(nodes::Vector{T}, d_nodes::Vector{T}, kernel::RK = RK_H1()) where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

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

Return: the prepared `NormalSpline` object.
"""
function prepare_interpolation(nodes::Vector{T},
                               d_nodes::Vector{T},
                               kernel::RK = RK_H1()
                              ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}

     es = ones(T, length(d_nodes))
     spline = _prepare_interpolation(Matrix(nodes'), Matrix(d_nodes'), Matrix(es'), kernel)
     return spline
end

"""
`interpolate(nodes::Vector{T}, values::Vector{T}, d_nodes::Vector{T}, d_values::Vector{T},
             kernel::RK = RK_H1())
             where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

prepare_interpolation and construct the 1D interpolating normal spline.
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

Return: the constructed `NormalSpline` object.
"""
function interpolate(nodes::Vector{T},
                     values::Vector{T},
                     d_nodes::Vector{T},
                     d_values::Vector{T},
                     kernel::RK = RK_H1()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
     es = ones(T, length(d_nodes))
     spline = _prepare_interpolation(Matrix(nodes'), Matrix(d_nodes'), Matrix(es'), kernel)
     spline = _construct_interpolation(spline, values, d_values)
     return spline
end
