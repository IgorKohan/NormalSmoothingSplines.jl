function _prepare_approximation(nodes_b::Matrix{T},
                                kernel::RK
                               ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    spline = _prepare_approximation(nothing, nodes_b, kernel)
    return spline
end

function _prepare_approximation(nodes::Union{Matrix{T}, Nothing},
                                nodes_b::Matrix{T},
                                kernel::RK
                               ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

     if isnothing(nodes)
         n_1 = 0
         n = size(nodes_b, 1)
     else
        n = size(nodes, 1)
        n_1 = size(nodes, 2)
        if n != size(nodes_b, 1)
            error("Matrices 'nodes' and 'nodes_b' have mismatched dimensions.")
        end
     end

     n_b = size(nodes_b, 1)
     n_1_b = size(nodes_b, 2)

     min_bound = Vector{T}(undef, n)
     compression::T = T(0.)
     maxx::T = T(0.)
     @inbounds for i = 1:n
         if !isnothing(nodes)
             min_bound[i] = nodes[i,1]
             maxx = nodes[i,1]
         else
             min_bound[i] = nodes_b[i,1]
             maxx = nodes_b[i,1]
         end
         for j = 1:n_1
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

     if !isnothing(nodes)
         t_nodes = similar(nodes)
     else
         t_nodes = nothing
     end
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

     if !isnothing(nodes)
         t_nodes_all = [t_nodes t_nodes_b]
     else
         t_nodes_all = t_nodes_b
     end

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
             error("Incorrect `kernel` type.")
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
                           -1
                          )
     return spline
end

function _construct_approximation(spline::NormalSpline{T, RK},
                                  values_lb::Vector{T},
                                  values_ub::Vector{T},
                                  maxiter::Int,
                                  ftol::T,
                                  cleanup::Bool = false
                                 ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    if !isnothing(spline._nodes)
         error("Spline object is not correctly prepared ('spline._nodes' is not empty).")
    end

    spline, nit_done = _construct_approximation(spline, T[], values_lb, values_ub, maxiter, ftol, cleanup)
    return spline, nit_done
end

function _construct_approximation(spline::NormalSpline{T, RK},
                                  values::Vector{T},
                                  values_lb::Vector{T},
                                  values_ub::Vector{T},
                                  maxiter::Int,
                                  ftol::T,
                                  cleanup::Bool = false
                                 ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    if isnothing(spline._nodes) && isnothing(spline._nodes_b)
        error("Spline object is not correctly prepared ('spline._nodes' is empty and 'spline._nodes_b' is empty).")
    end
    if isnothing(spline._nodes_b)
        error("Spline object is not correctly prepared ('spline._nodes_b' is empty).")
    end

    m1 = 0
    if !isnothing(spline._nodes)
        m1 = size(spline._nodes, 2)
    end
    m2 = size(spline._nodes_b, 2)

    if length(values) != m1
        error("Number of 'values' does not correspond to the number of interpolating nodes ('spline._nodes').")
    end
    if length(values_lb) != m2
        error("Number of 'values_lb' does not correspond to the number of approximating nodes ('spline._nodes_b').")
    end
    if length(values_ub) != m2
        error("Number of 'values_ub' does not correspond to the number of approximating nodes.")
    end

    if length(values_ub[values_ub .< values_lb]) > 0
        error("Incorrect bounds: 'values_ub' are less than 'values_lb'.")
    end

    if isnothing(spline._chol)
        error("Gram matrix was not factorized.")
    end

    umax = T(0.)
    @inbounds for i = 1:m2
        val = abs(values_lb[i])
        if val != Inf && val > umax
            umax = val
        end
        val = abs(values_ub[i])
        if val != Inf && val > umax
            umax = val
        end
    end

    values_b = mu = zeros(T, m2)
    incorrect_bounds::Bool = false
    @inbounds for i = 1:m2
        if abs(values_lb[i]) == Inf && abs(values_ub[i]) == Inf
            incorrect_bounds = true
            break
        end
        if abs(values_lb[i]) != Inf && abs(values_ub[i]) != Inf
            values_b[i] = (values_lb[i] + values_ub[i]) / T(2.0)
        end
        if abs(values_lb[i]) == Inf
            values_b[i] = values_ub[i] - umax
        end
        if abs(values_ub[i]) == Inf
            values_b[i] = values_lb[i] + umax
        end
    end
    if incorrect_bounds
        error("Incorrect_bounds. Both 'values_ub[i]' and 'values_lb[i]' are set to Inf.")
    end

    if m1 > 0
        values_all = [values; values_b]
    else
        values_all = values_b
    end

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
                          -1
                         )

    eps = T(1.e-10)
    spline, nit_done = _qp(spline, Int[], maxiter, ftol, eps, cleanup)
    return spline, nit_done
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
        error("Spline was not constructed.")
    end

    if isnothing(spline._ier < 0)
        error("Spline coefficients are not calculated.")
    end

    if size(points, 1) != size(spline._nodes_b, 1)
        if size(points, 1) == 1 && size(points, 2) > 1
            error("Incorrect first dimension of the `points` parameter (use 'evaluate_at' function for evaluating the spline at one point).")
        else
            error("Incorrect first dimension of the `points` parameter (the spline was built in the space of different dimension).")
        end
    end

    if isnothing(spline._nodes)
       n_1 = 0
    else
       n_1 = size(spline._nodes, 2)
    end

    n = size(spline._nodes_b, 1)
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

    if spline._ier != 0 #  optimal solution not found

        h_values_b = Vector{T}(undef, n_1_b)
        @inbounds for p = 1:m
            for i = 1:n_1_b
                h_values_b[i] = _rk(spline._kernel, pts[:,p], spline._nodes_b[:,i])
            end
            if n_1 > 0
                for i = 1:n_1
                    h_values[i] = _rk(spline._kernel, pts[:,p], spline._nodes[:,i])
                end
                spline_values[p] = sum(spline._mu .* [h_values; h_values_b; -h_values_b])
            else
                spline_values[p] = sum(spline._mu .* [h_values_b; -h_values_b])
            end
        end
    else # optimal solution found
        ak = spline._active[spline._active .!= 0]
        nak = length(ak)
        h_values_b = Vector{T}(undef, nak)
        mu_b = Vector{T}(undef, nak)
        @inbounds for p = 1:m
            spline_values[p] = T(0.)
            if n_1 > 0
                for i = 1:n_1
                    h_values[i] = _rk(spline._kernel, pts[:,p], spline._nodes[:,i])
                end
                spline_values[p] += sum(spline._mu[1:n_1] .* h_values)
            end

            for i = 1:nak
                ii = ak[i]
                if ii > 0
                    mu_b[i] = spline._mu[ii+n_1]
                    h_values_b[i] = _rk(spline._kernel, pts[:,p], spline._nodes_b[:,ii])
                    s = 0
                else
                    ii = -ii
                    mu_b[i] = spline._mu[ii+n_1+n_1_b]
                    h_values_b[i] = -_rk(spline._kernel, pts[:,p], spline._nodes_b[:,ii])
                    s = 0
                end
            end
            spline_values[p] += sum(mu_b .* h_values_b)
        end
    end # if spline._ier != 0

    return spline_values
end
