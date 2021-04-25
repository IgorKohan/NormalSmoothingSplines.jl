function _prepare_approximation(nodes::Matrix{T},
                                nodes_b::Matrix{T},
                                kernel::RK
                               ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
     n = size(nodes, 1)
     n_1 = size(nodes, 2)

     n_b = size(nodes_b, 1)
     n_1_b = size(nodes_b, 2)
     # TODO test n == n_b

     min_bound = Vector{T}(undef, n)
     compression::T = 0
     @inbounds for i = 1:n
         min_bound[i] = nodes[i,1]
         maxx::T = nodes[i,1]
         for j = 2:n_1
             min_bound[i] = min(min_bound[i], nodes[i,j])
             maxx = max(maxx, nodes[i,j])
         end
         for j = 1:n_1_b
             min_bound[i] = min(min_bound[i], nodes_b[i,j])
             maxx = max(maxx, nodes_b[i,j])
         end
         compression = max(compression, maxx - min_bound[i])
     end

     if compression <= eps(T(1.0))
         error("Cannot prepare the spline: `nodes` data are not correct.")
     end

     t_nodes = similar(nodes)
     @inbounds for j = 1:n_1
         for i = 1:n
             t_nodes[i,j] = (nodes[i,j] - min_bound[i]) / compression
         end
     end
     t_nodes_b = similar(nodes_b)
     @inbounds for j = 1:n_1_b
         for i = 1:n
             t_nodes_b[i,j] = (nodes_b[i,j] - min_bound[i]) / compression
         end
     end

     t_nodes_all = [t_nodes t_nodes_b]

     if T(kernel.ε) == T(0.0)
         ε = _estimate_ε(t_nodes_all)
         if isa(kernel, RK_H0)
             kernel = RK_H0(ε)
         elseif isa(kernel, RK_H1)
             ε *= T(1.5)
             kernel = RK_H1(ε)
         elseif isa(kernel, RK_H2)
             ε *= T(2.0)
             kernel = RK_H2(ε)
         else
             error("incorrect `kernel` type.")
         end
     end

     gram = _gram(t_nodes_all, kernel)
     chol = nothing
     try
         chol = cholesky(gram)
     catch
         error("Cannot prepare the spline: Gram matrix is degenerate.")
     end

     cond = _estimate_cond(gram, chol)

     spline = NormalSpline(kernel,
                           compression,
                           t_nodes,
                           t_nodes_b,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           nothing,
                           min_bound,
                           gram,
                           chol,
                           nothing,
                           nothing,
                           cond,
                           0
                          )
     return spline
end

function _construct_approximation(spline::NormalSpline{T, RK},
                                  values::Vector{T},
                                  values_lb::Vector{T},
                                  values_ub::Vector{T},
                                  nit::Int,
                                  cleanup::Bool = false
                                 ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    if(length(values) != size(spline._nodes, 2))
        error("Number of data values does not correspond to the number of nodes.")
    end
    if isnothing(spline._chol)
        error("Gram matrix was not factorized.")
    end

# TODO: check if the interpolating spline is the solution
#       values_lb <= values <= values_ub
#       if yes then do not smooth
#..

    values_b = similar(values_lb)
    values_b .= (values_lb .+ values_ub) ./ T(2.0)
    values_all = [values; values_b]
    mu = Vector{T}(undef, size(spline._gram, 1))
    ldiv!(mu, spline._chol, values_all)

    spline = NormalSpline(spline._kernel,
                          spline._compression,
                          spline._nodes,
                          spline._nodes_b,
                          values,
                          values_lb,
                          values_ub,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          nothing,
                          spline._min_bound,
                          spline._gram,
                          spline._chol,
                          mu,
                          nothing,
                          spline._cond,
                          0
                         )

    spline = _qp1(spline, nit, T(1.e-10), true, cleanup)
    return spline
end

###################

function _evaluate(spline::NormalSpline{T, RK}, points::Matrix{T},
                  ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    if !isnothing(spline._nodes_b) || !isnothing(spline._d_nodes_b)
        _evaluate_approximation(spline, points)
    else
        _evaluate_interpolation(spline, points)
    end
end

function _evaluate_gradient(spline::NormalSpline{T, RK},
                            point::Vector{T}
                           ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    if !isnothing(spline._nodes_b) || !isnothing(spline._d_nodes_b)
        _evaluate_approximation_gradient(spline, point)
    else
        _evaluate_interpolation_gradient(spline, point)
    end
end

function _evaluate_approximation_gradient(spline::NormalSpline{T, RK},
                                          point::Vector{T}
                                         ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    # TODO - implement                                         
    error("_evaluate_approximation_gradient: Not implemented.")
end

function _evaluate_approximation(spline::NormalSpline{T, RK}, points::Matrix{T},
                                ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}
    if isnothing(spline)
        error("Spline was not prepared.")
    end
    if isnothing(spline._mu)
        error("Spline coefficients were not calculated.")
    end

    # TODO dimenion checks
    # if size(points, 1) != size(spline._nodes, 1)
    #     if size(points, 1) == 1 && size(points, 2) > 1
    #         error("Incorrect first dimension of the `points` parameter (use 'evaluate_one' function for evaluating the spline at one point).")
    #     else
    #         error("Incorrect first dimension of the `points` parameter (the spline was built in the space of different dimension).")
    #     end
    # end

    n = size(spline._nodes, 1)
    n_1 = size(spline._nodes, 2)
    n_1_b = size(spline._nodes_b, 2)
    m = size(points, 2)

    pts = similar(points)
    @inbounds for j = 1:m
        for i = 1:n
            pts[i,j] = (points[i,j] - spline._min_bound[i]) / spline._compression
        end
    end

    spline_values = Vector{T}(undef, m)
    h_values = Vector{T}(undef, n_1)
    h_values_b = Vector{T}(undef, n_1_b)

    # COH TODO - use only active constraints
    @inbounds for p = 1:m
        for i = 1:n_1
            h_values[i] = _rk(spline._kernel, pts[:,p], spline._nodes[:,i])
        end
        for i = 1:n_1_b
            h_values_b[i] = _rk(spline._kernel, pts[:,p], spline._nodes_b[:,i])
        end
        spline_values[p] = sum(spline._mu .* [h_values; h_values_b; -h_values_b])
    end

    return spline_values
end
