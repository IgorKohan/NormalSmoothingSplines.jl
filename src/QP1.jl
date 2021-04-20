function qp1(spline::NormalSpline{T, RK}, nit::Int, cleanup::Bool
            ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    path = "0_qp.log"
    if ispath(path)
       rm(path)
    end
    open(path,"a") do io
        println(io,"qp started.\r\n")
    end

# Initialization..
    n = size(spline._nodes, 1)
    n_1 = size(spline._nodes, 2)
    n_1_b = size(spline._nodes_b, 2)
    L = n_1
    M = n_1_b
    S = L + M
    M2 = 2*M
    N = L + M2

    b = [spline._values; spline._values_ub; -spline._values_lb ]

    ak = zeros(Int32, S)
    pk = zeros(Int32, M2) # M
    ek = zeros(T, M2)     # M ???

    @inbounds for j = 1:L
        ak[j] = j
    end

    @inbounds for j = 1:M2 # M ????
        pk[j] = j + L
    end

    mu = zeros(T, S) #?? N
    @inbounds for j = 1:S
        mu[j] = spline._mu[j]
    end

# first case of constructing feasible point
    nak = L
    npk = M2
#..

# Main cycle
    nit = 2 # Debugging
    @inbounds for it = 1:nit

# Calculating lambda
        lambda = zeros(T, S) # ?? N
        if nak > 0
            mat = Matrix{T}(undef, nak, nak)
            w = Vector{T}(undef, nak)
            @inbounds for j = 1:nak
                  jj = ak[j]
                  w[j] = b[jj]
                  for i = 1:j
                      ii = ak[i]
                      mat[i,j] = spline._gram[ii, jj]
                      mat[j,i] = mat[i,j]
                  end
            end
            ldiv!(cholesky!(mat), w)
#            cholesky!(mat)
#            w = mat \ w
            @inbounds for j = 1:nak
                  jj = ak[j]
                  lambda[jj] = w[j]
            end
        end
#..Calculating lambda

        if nak == S
           @goto STEP5
        end

# Calculating ek
        @inbounds for j = 1:npk
#..

        end
#.. Calculating ek




@label STEP5



    end
#..Main cycle

    spline = NormalSpline(spline._kernel,
                  spline._compression,
                  spline._nodes,
                  spline._nodes_b,
                  spline._values,
                  spline._values_lb,
                  spline._values_ub,
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
                  spline._mu,
                  spline._cond,
                  0
                 )

    open(path,"a") do io
         println(io,"\r\nqp completed.")
    end
    return spline
end

# open("path,"a") do io
#     @printf io "model_id type_of_samples  n_of_samples  type_of_kernel  regular_grid_size\n"
#     @printf io "separation_distance:%0.1e  fill_distance:%0.1e fs_ratio:%0.1e\n" separation_distance fill_distance fs_ratio
#     @printf io "F_MIN:%0.1e  F_MAX:%0.1e \n" f_min f_max
#     @printf io "%2d      %2d             %4d             %1d               %3d\n" model_id type_of_samples n_of_samples type_of_kernel regular_grid_size
#     @printf io "RMSE: %0.1e  MAE:%0.1e RRMSE: %0.1e RMAE:%0.1e  SPLINE_MIN:%0.1e  SPLINE_MAX:%0.1e   EPS:%0.1e   COND: %0.1e\n" rmse mae rmse rmae spline_min spline_max ε cond
#     @printf io "c_time: %0.1e  e_time: %0.1e\n\n" c_time e_time
# end
