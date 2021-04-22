function _qp1(spline::NormalSpline{T, RK}, nit::Int, logging::Bool = true, cleanup::Bool = false
             ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    if logging == true
        path = "0_qp1.log"
        if ispath(path)
           rm(path)
        end
        open(path,"a") do io
            println(io,"_qp1 started.\r\n")
        end
    end

# Initialization..
    m1 = size(spline._nodes, 2)
    m2 = size(spline._nodes_b, 2)
    m1p1 = m1 + 1
    m2pm2 = m2 + m2
    m = m1 + m2
    n = m1 + m2 + m2
    b = [spline._values; spline._values_ub; -spline._values_lb ]

    # first case of constructing feasible point
    nak = m1
    npk = m2pm2
    ak = zeros(Int32, m)
    pk = zeros(Int32, npk)
    ek = zeros(T, m)

    @inbounds for j = 1:nak
        ak[j] = j
    end

    @inbounds for j = 1:npk
        pk[j] = j + m1
    end

    #.. first case of constructing feasible point
    mu = zeros(T, n)
    @inbounds for j = 1:m
        mu[j] = spline._mu[j]
    end

    f_add::Bool = false
    f_del::Bool = false

# Main cycle
    nit = 2 # Debugging
    nit_done = 0
    @inbounds for it = 1:nit
        nit_done += 1
        lambda = zeros(T, n)

        if nak > 0
#  Calculating Gram matrix factorization and lambda
            w = Vector{T}(undef, nak)
            @inbounds for j = 1:nak
                jj = ak[j]
                w[j] = b[jj]
            end

            if !f_add && !f_del
                mat = Matrix{T}(undef, nak, nak)
                si = T(1.)
                sj = T(1.)
                @inbounds for j = 1:nak
                      jj = ak[j]
                      if jj > m
                          jj -= m2
                          sj = T(-1.)
                      else
                          sj = T(1.)
                      end
                      for i = 1:j
                          ii = ak[i]
                          if ii > m
                              ii -= m2
                              si = T(-1.)
                          else
                              si = T(1.)
                          end
                          mat[i,j] = si * sj * spline._gram[ii, jj]
                          mat[j,i] = mat[i,j]
                      end
                end # @inbounds for j = 1:nak

                try
                    cholesky!(mat)
                catch
                    error("_qp1: Gram matrix is degenerate.")
                end

            end #if !f_add && !f_del

            if f_del

            end # if f_del

            if f_add

            end #if f_add

            #  Calculating lambda
            w = mat \ w

            @inbounds for j = 1:nak
                  jj = ak[j]
                  lambda[jj] = w[j]
            end
            #..Calculating lambda

        end #if nak > 0
#..Calculating Gram matrix factorization and lambda

        l_m = zeros(T, n)
        l_m .=  lambda .- mu

        if nak == m
           #@goto STEP7
        end

# Calculating ek
        @inbounds for j = 1:npk
#..

        end
#.. Calculating ek


@label STEP1


@label STEP7



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
                  nit_done
                 )

    if logging == true
        open(path,"a") do io
             println(io,"\r\n_qp1 completed.")
        end
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
