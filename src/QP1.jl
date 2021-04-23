function _qp1(spline::NormalSpline{T, RK}, nit::Int, eps::T = T(1.e-10),
              logging::Bool = true, cleanup::Bool = false
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
    b = [spline._values; spline._values_ub; -spline._values_lb]

    m1p1 = m1 + 1
    m2pm2 = m2 + m2
    m = m1 + m2
    n = m1 + m2 + m2

    # first case of constructing feasible point
    nak = m1
    npk = m2pm2
    ak = zeros(Int32, m)
    pk = zeros(Int32, npk)
    ek = ones(T, npk)  # TODO DEBUG

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
    lambda = zeros(T, n)
    l_m = zeros(T, n)
    w = zeros(T, m)
    t_max::T = typemax(T)

    f_add::Bool = false
    f_del::Bool = false
    f_opt::Bool = false
    i_add::Int = 0
    i_del::Int = 0

# Main cycle
    nit = 2 # Debugging
    nit_done = 0
    mat = Matrix{T}(undef, nak, nak)
#
    @inbounds for it = 1:nit
        nit_done += 1

        @inbounds for i = 1:n
            lambda[i] = T(0.)
        end

        if nak > 0
#  Calculating Gram matrix factorization and lambda
            @inbounds for j = 1:nak
                jj = ak[j]
                w[j] = b[jj]
            end

            f_add = false # TODO DEBUGGING
            f_del = false # TODO DEBUGGING

            if !f_add && !f_del
                # if it > 1
                #     error("_qp1: !f_add && !f_del.") # TODO DEBUGGING
                # end
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
            w = mat \ w[1:nak]

            @inbounds for j = 1:nak
                  jj = ak[j]
                  lambda[jj] = w[j]
            end
            #..Calculating lambda

        end #if nak > 0
#..Calculating Gram matrix factorization and lambda

        if nak == m && nak > m1
            @inbounds for i = 1:n
                mu[i] = lambda[i]
            end

            f_opt = true
            @inbounds for i = 1:nak
                  ii = ak[i]
                  if ii <= m1
                      continue #.. for i = 1:nak
                  end
                  if lambda[ii] < T(eps)
                      continue #.. for i = 1:nak
                  end
                  f_opt = false
                  i_del = ii
                  npk += 1
                  pk[npk] = ii
                  ak = deleteat!(ak, i)
                  nak -= 1
                  f_del = true
                  break     #.. for i = 1:nak
            end #.. for i = 1:nak

            if f_opt == true
                break #..Main cycle
            end

            continue #..Main cycle
       end # nak == m

        @inbounds for i = 1:n
            l_m[i] = lambda[i] - mu[i]
        end

# Calculating t_min, i_min
        ek = ones(T, npk) # TODO DEL DEBUG
        si = T(1.)
        sj = T(1.)
        t_min::T = t_max
        i_min::Int = 0
        @inbounds for i = 1:npk
            eik::T = T(0.)
            s::T = T(0.)
            ii = pk[i]
            if ii > m
                ii -= m2
                si = T(-1.)
            else
                si = T(1.)
            end
            @inbounds for j = 1:n
                jj = j
                if jj > m
                    jj -= m2
                    sj = T(-1.)
                else
                    sj = T(1.)
                end
                eik += l_m[j] * si * sj * spline._gram[ii, jj]
                s += mu[j] * si * sj * spline._gram[ii, jj]
            end #.. for j = 1:n

            ek[i] = eik # TODO DEL DEBUG
            if eik > T(eps)
                tik = (b[ii] - s) / eik
                if tik < t_min
                    i_min = ii
                    t_min = tik
                end
            end
        end #.. for i = 1:npk
#.. Calculating t_min, i_min

        if t_min < (T(1.0) + eps) # the projection is not feasible
            if t_min < T(0.)
                 t_min = T(0.)
            end
            @inbounds for i = 1:n
                mu[i] += t_min * (lambda[i] - mu[i])
            end
            f_add = true
            nak += 1
            ak[nak] = i_min
            pk = deleteat!(pk, findall(x->x==i_min, pk))
            npk -= 1
        else                      # the projection is feasible
            @inbounds for i = 1:n
                mu[i] = lambda[i]
            end

            if nak == m1
                f_opt = true
                break             #..Main cycle
            end

            f_opt = true
            @inbounds for i = 1:nak
                  ii = ak[i]
                  if ii <= m1
                      continue #.. for i = 1:nak
                  end
                  if lambda[ii] < T(eps)
                      continue #.. for i = 1:nak
                  end
                  f_opt = false
                  i_del = ii
                  npk += 1
                  pk[npk] = ii
                  ak = deleteat!(ak, i)
                  nak -= 1
                  f_del = true
                  break     #.. for i = 1:nak
            end #.. for i = 1:nak

            if f_opt == true
                break #..Main cycle
            end

            continue #..Main cycle
        end #.. t_min < (T(1.0) + eps)

    end #..Main cycle

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
