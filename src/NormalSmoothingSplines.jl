module NormalSmoothingSplines

#### Approximating splines inteface definition.
export prepare_approximation, construct_approximation, approximate
export evaluate_approximation, evaluate_approximation_at #, evaluate_gradient
export NormalSpline, RK_H0, RK_H1, RK_H2
export get_epsilon, estimate_epsilon, get_cond, estimate_cond
export estimate_accuracy
# -- 1D case --
#export evaluate_derivative
# --
###
#include("./examples/Main.jl")
###

using LinearAlgebra

abstract type ReproducingKernel end
abstract type ReproducingKernel_0 <: ReproducingKernel end
abstract type ReproducingKernel_1 <: ReproducingKernel_0 end
abstract type ReproducingKernel_2 <: ReproducingKernel_1 end

abstract type AbstractSpline end

@doc raw"
`struct NormalSpline{T, RK} <: AbstractSpline where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Define a structure containing full information of a normal spline
# Fields
- `_kernel`: a reproducing kernel spline was built with
- `_compression`: factor of transforming the original node locations into unit hypercube
- `_nodes`: transformed function value interpolation nodes
- `_nodes_b`: transformed function value approximation nodes
- `_values`: function values at interpolation nodes
- `_values_lb`: function lower bound values at approximation nodes
- `_values_ub`: function upper bound values at approximation nodes
- `_d_nodes`: transformed function directional derivative interpolation nodes
- `_d_nodes_b`: transformed function directional derivative approximation nodes
- `_es`: normalized derivative directions at interpolation nodes
- `_es_b`: normalized derivative directions at approximation nodes
- `_d_values`: function directional derivative values at interpolation nodes
- `_d_values_lb`: function lower bound directional derivative values at approximation nodes
- `_d_values_ub`: function upper bound directional derivative values at approximation nodes
- `_min_bound`: minimal bounds of the original node locations area
- `_gram`: Gram matrix of the problem
- `_chol`: Cholesky factorization of the Gram matrix
- `_mu`: spline coefficients
- `_active`: active inequality constraint numbers at solution
- `_cond`: estimation of the Gram matrix condition number
- `_iter`: number of the qp algorithm iterations done
"
struct NormalSpline{T, RK} <: AbstractSpline where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    _kernel::RK
    _compression::T
    _nodes::Union{Matrix{T}, Nothing}
    _nodes_b::Union{Matrix{T}, Nothing}
    _values::Union{Vector{T}, Nothing}
    _values_lb::Union{Vector{T}, Nothing}
    _values_ub::Union{Vector{T}, Nothing}
    _d_nodes::Union{Matrix{T}, Nothing}
    _d_nodes_b::Union{Matrix{T}, Nothing}
    _es::Union{Matrix{T}, Nothing}
    _es_b::Union{Matrix{T}, Nothing}
    _d_values::Union{Vector{T}, Nothing}
    _d_values_lb::Union{Vector{T}, Nothing}
    _d_values_ub::Union{Vector{T}, Nothing}
    _min_bound::Union{Vector{T}, Nothing}
    _gram::Union{Matrix{T}, Nothing}
    _chol::Union{Cholesky{T, Matrix{T}}, Nothing}
    _mu::Union{Vector{T}, Nothing}
    _active::Union{Vector{Int}, Nothing}
    _cond::T
    _iter::Int
end

include("./NormalInterpolatingSplines.jl")
include("./ReproducingKernels.jl")
include("./GramMatrix.jl")
include("./Utils.jl")
include("./QP1.jl")
include("./Interpolate.jl")
include("./Approximate.jl")

"""
`prepare_approximation(nodes::Matrix{T}, nodes_b::Matrix{T}, kernel::RK = RK_H0())
                       where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare the approximating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value interpolation nodes. It means that each column in the matrix defines one node.
- `nodes_b`: function value approximation nodes.
          This should be an `n×n_1_b` matrix, where `n` is dimension of the sampled space and
          `n_1_b` is the number of function value approximation nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: prepared `NormalSpline` object.
"""
function prepare_approximation(nodes::Matrix{T},
                               nodes_b::Matrix{T},
                               kernel::RK = RK_H0()
                              ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_approximation(nodes, nodes_b, kernel)
     return spline
end

"""
`construct_approximation(spline::NormalSpline{T, RK},
                            values::Vector{T}, values_lb::Vector{T}, values_ub::Vector{T}, nit::Int)
                            where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

construct the approximating normal spline by calculating its coefficients and
completely initializing the `NormalSpline` object.
# Arguments
- `spline`: the partly initialized `NormalSpline` object returned by `prepare` function.
- `values`: function values at interpolation nodes.
- `values_lb`: function lower bound values at approximation nodes
- `values_ub`: function upper bound values at approximation nodes
- `nit`: maximum number of algorithm iterations

Return: constructed `NormalSpline` object.
"""
function construct_approximation(spline::NormalSpline{T, RK},
                                    values::Vector{T},
                                    values_lb::Vector{T},
                                    values_ub::Vector{T},
                                    nit::Int = 100
                                   ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    spline = _construct_approximation(spline, values, values_lb, values_ub, nit)
    return spline
end

"""
`approximate(nodes::Matrix{T}, values::Vector{T}, nodes_b::Matrix{T}, values_lb::Vector{T}, values_ub::Vector{T},
             nit::Int, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare and construct the approximating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `nodes_b`: function value approximation nodes.
        This should be an `n×n_1_b` matrix, where `n` is dimension of the sampled space and
        `n_1_b` is the number of function value approximation nodes. It means that each column in the matrix defines one node.
- `values`: function values at interpolation nodes.
- `values_lb`: function lower bound values at approximation nodes
- `values_ub`: function upper bound values at approximation nodes
- `nit`: maximum number of algorithm iterations
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: constructed `NormalSpline` object.
"""
function approximate(nodes::Matrix{T},
                     values::Vector{T},
                     nodes_b::Matrix{T},
                     values_lb::Vector{T},
                     values_ub::Vector{T},
                     nit::Int = 100,
                     kernel::RK = RK_H0()
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_approximation(nodes, nodes_b, kernel)
     spline = _construct_approximation(spline, values, values_lb, values_ub, nit)
     return spline
end

"""
`evaluate_approximation(spline::NormalSpline{T, RK}, points::Matrix{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the approximating spline values at `points` locations.

# Arguments
- `spline: constructed `NormalSpline` object.
- `points`: locations at which spline values are evaluating.
            This should be an `n×m` matrix, where `n` is dimension of the sampled space
            and `m` is the number of locations where spline values are evaluating.
            It means that each column in the matrix defines one location.

Return: `Vector{T}` of the spline values at the locations defined in `points`.
"""
function evaluate_approximation(spline::NormalSpline{T, RK}, points::Matrix{T}
                               ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate_approximation(spline, points)
end

"""
`evaluate_approximation_at(spline::NormalSpline{T, RK}, point::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the approximating spline value at the `point` location.

# Arguments
- `spline`: constructed `NormalSpline` object.
- `point`: location at which spline value is evaluating.
           This should be a vector of size `n`, where `n` is dimension of the sampled space.

Return: the spline value at the location defined in `point`.
"""
function evaluate_approximation_at(spline::NormalSpline{T, RK}, point::Vector{T}
                                  ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate_approximation(spline, reshape(point, :, 1))[1]
end

########

"""
`get_epsilon(spline::NormalSpline{T, RK}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Get the 'scaling parameter' of Bessel Potential space the spline was built in.
# Arguments
- `spline`: prepared `NormalSpline` object.

Return: `ε` - the 'scaling parameter'.
"""
function get_epsilon(spline::NormalSpline{T, RK}
                    ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return spline._kernel.ε
end

"""
`estimate_epsilon(nodes::Matrix{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Get the estimation of the 'scaling parameter'2 of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
           It must be a struct object of the following type:
             `RK_H0` if the spline is constructing as a continuous function,
             `RK_H1` if the spline is constructing as a differentiable function,
             `RK_H2` if the spline is constructing as a twice differentiable function.
Return: estimation of `ε`.
"""
function estimate_epsilon(nodes::Matrix{T},
                          kernel::RK = RK_H0()
                         ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    ε = _estimate_epsilon(nodes, kernel)
    return ε
end

"""
`estimate_epsilon(nodes::Matrix{T}, d_nodes::Matrix{T}, kernel::RK = RK_H1()) where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space
           and `n_1` is the number of function value nodes.
           It means that each column in the matrix defines one node.
- `d_nodes`: function directional derivative nodes.
           This should be an `n×n_2` matrix, where `n` is dimension of the sampled space and
           `n_2` is the number of function directional derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline will be constructed in.
            It must be a struct object of the following type:
            `RK_H1` if the spline is constructing as a differentiable function,
            `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
function estimate_epsilon(nodes::Matrix{T},
                          d_nodes::Matrix{T},
                          kernel::RK = RK_H1()
                         ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
    ε = _estimate_epsilon(nodes, d_nodes, kernel)
    return ε
end

"""
`estimate_cond(spline::NormalSpline{T, RK}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Get an estimation of the Gram matrix condition number. It needs the `spline` object is prepared and requires O(N^2) operations.
(C. Brás, W. Hager, J. Júdice, An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP Vol.20, No.3, 2012.)
# Arguments
- `spline`: prepared `NormalSpline` object.

Return: an estimation of the Gram matrix condition number.
"""
function estimate_cond(spline::NormalSpline{T, RK}
                      ) where {T <: AbstractFloat, RK <: ReproducingKernel}
    return spline._cond
end

"""
`estimate_accuracy(spline::NormalSpline{T, RK}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Assess accuracy of interpolation results by analyzing residuals.
# Arguments
- `spline`: constructed `NormalSpline` object.

Return: an estimation of the number of significant digits in the interpolation result.
"""
function estimate_accuracy(spline::NormalSpline{T, RK}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _estimate_accuracy(spline)
end

############################## One-dimensional case

"""
`prepare_approximation(nodes::Vector{T}, nodes_b::Vector{T}, kernel::RK = RK_H0())
                       where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare the 1D approximating normal spline by constructing and factoring a Gram matrix of the problem.
Initialize the `NormalSpline` object.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n_1` vector where `n_1` is the number of function value interpolation nodes.
- `nodes_b`: function value approximation nodes.
           This should be an `n_1_b` vector where `n_1_b` is the number of function value approximation nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: prepared `NormalSpline` object.
"""
function prepare_approximation(nodes::Vector{T},
                               nodes_b::Vector{T},
                               kernel::RK = RK_H0()
                              ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_approximation(Matrix(nodes'), Matrix(nodes_b'), kernel)
     return spline
end

"""
`approximate(nodes::Vector{T}, values::Vector{T}, nodes_b::Vector{T}, values_lb::Vector{T}, values_ub::Vector{T},
             nit::Int, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Prepare and construct the 1D approximating normal spline.
# Arguments
- `nodes`: function value interpolation nodes.
          This should be an `n_1` vector where `n_1` is the number of function value interpolation nodes.
- `values`: function values at interpolation nodes.
- `nodes_b`: function value approximation nodes.
          This should be an `n_1_b` vector where `n_1_b` is the number of function value approximation nodes.
- `values_lb`: function lower bound values at approximation nodes
- `values_ub`: function upper bound values at approximation nodes
- `nit`: maximum number of algorithm iterations
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: constructed `NormalSpline` object.
"""
function approximate(nodes::Vector{T},
                     values::Vector{T},
                     nodes_b::Vector{T},
                     values_lb::Vector{T},
                     values_ub::Vector{T},
                     nit::Int = 100,
                     kernel::RK = RK_H0()
               ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     spline = _prepare_approximation(Matrix(nodes'), Matrix(nodes_b'), kernel)
     spline = _construct_approximation(spline, values, values_lb, values_ub, nit)
     return spline
end

"""
`evaluate_approximation(spline::NormalSpline{T, RK}, points::Vector{T}) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the 1D approximating spline values at the `points` locations.

# Arguments
- `spline`: constructed `NormalSpline` object.
- `points`: locations at which spline values are evaluating.
            This should be a vector of size `m` where `m` is the number of evaluating points.

Return: spline value at the `point` location.
"""
function evaluate_approximation(spline::NormalSpline{T, RK}, points::Vector{T}
                               ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    return _evaluate_approximation(spline, Matrix(points'))
end

"""
`evaluate_approximation_at(spline::NormalSpline{T, RK}, point::T) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Evaluate the 1D approximating spline value at the `point` location.

# Arguments
- `spline`: constructed `NormalSpline` object.
- `point`: location at which spline value is evaluating.

Return: spline value at the `point` location.
"""
function evaluate_approximation_at(spline::NormalSpline{T, RK},
                                       point::T
                                      ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    v_points = Vector{T}(undef, 1)
    v_points[1] = point
    return _evaluate_approximation(spline, Matrix(v_points'))[1]
end

"""
`estimate_epsilon(nodes::Vector{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: function value interpolation nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
function estimate_epsilon(nodes::Vector{T},
                          kernel::RK = RK_H0()
                         ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    ε = _estimate_epsilon(Matrix(nodes'), kernel)
    return ε
end

"""
`estimate_epsilon(nodes::Vector{T}, d_nodes::Vector{T}, kernel::RK = RK_H1()) where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Get an the estimation of the 'scaling parameter' of Bessel Potential space the spline being built in.
It coincides with the result returned by `get_epsilon` function.
# Arguments
- `nodes`: function value interpolation nodes.
- `d_nodes`: function derivative nodes.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: estimation of `ε`.
"""
function estimate_epsilon(nodes::Vector{T},
                          d_nodes::Vector{T},
                          kernel::RK = RK_H1()
                         ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
    ε = _estimate_epsilon(Matrix(nodes'), Matrix(d_nodes'), kernel)
    return ε
end

"""
`get_cond(nodes::Matrix{T}, kernel::RK = RK_H0()) where {T <: AbstractFloat, RK <: ReproducingKernel_0}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.
# Arguments
- `nodes`: function value interpolation nodes.
           This should be an `n×n_1` matrix, where `n` is dimension of the sampled space and
           `n_1` is the number of function value nodes. It means that each column in the matrix defines one node.
- `kernel`: reproducing kernel of Bessel potential space the normal spline is constructed in.
            It must be a struct object of the following type:
              `RK_H0` if the spline is constructing as a continuous function,
              `RK_H1` if the spline is constructing as a differentiable function,
              `RK_H2` if the spline is constructing as a twice differentiable function.

Return: a value of the Gram matrix spectral condition number.
"""
function get_cond(nodes::Matrix{T}, kernel::RK) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
  return _get_cond(nodes, kernel)
end

"""
`get_cond(nodes::Matrix{T}, d_nodes::Matrix{T}, es::Matrix{T}, kernel::RK = RK_H1()) where {T <: AbstractFloat, RK <: ReproducingKernel_1}`

Get a value of the Gram matrix spectral condition number. It is obtained by means of the matrix SVD decomposition and requires ``O(N^3)`` operations.
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

Return: a value of the Gram matrix spectral condition number.
"""
function get_cond(nodes::Matrix{T}, d_nodes::Matrix{T}, es::Matrix{T}, kernel::RK = RK_H1()
                 ) where {T <: AbstractFloat, RK <: ReproducingKernel_1}
    return _get_cond(nodes, d_nodes, es, kernel)
end


end # module
