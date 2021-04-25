function _qp1(spline::NormalSpline{T, RK}, nit::Int, eps::T,
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
    ak = zeros(Int, m)
    pk = zeros(Int, npk)

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
    t_max::T = typemax(T)

    f_add::Bool = false
    f_del::Bool = false
    f_opt::Bool = false
    i_add = 0
    i_del = 0

# Main cycle
    mat = Matrix{T}(undef, nak, nak)
    nit_done = 0
#
    @inbounds for it = 1:nit
        nit_done += 1

        @inbounds for i = 1:n
            lambda[i] = T(0.)
        end

        w = zeros(T, nak)
        @inbounds for j = 1:nak
            jj = ak[j]
            w[j] = b[jj]
        end

        if nak > 0

            mat = Matrix{T}(undef, nak, nak) # TODO delete DEBUGGING

#  Calculating Gram matrix factorization and lambda

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
                    mat = cholesky!(mat)
                catch
                    error("_qp1: Gram 'mat' matrix is degenerate.")
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
                  f_del = true
                  i_del = ii
                  npk += 1
                  pk[npk] = ii
                  ip1 = i + 1
                  @inbounds for k = ip1:nak
                      ak[k-1] = ak[k]
                  end
                  ak[nak] = 0
                  nak -= 1
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
        # ek = ones(T, npk) # TODO DEL DEBUG
        # tk = zeros(T, npk) # TODO DEL DEBUG
        si = T(1.)
        sj = T(1.)
        t_min::T = t_max
        i_min = 0
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

            # ek[i] = eik # TODO DEL DEBUG
            # tk[i] = (b[pk[i]] - s) / eik # TODO DEL DEBUG
            if eik > T(eps)
                ii = pk[i]
                tik = (b[ii] - s) / eik
                if tik < t_min
                    i_min = ii
                    t_min = tik
                end
            end
        end #.. for i = 1:npk
#.. Calculating t_min, i_min

        if t_min < (T(1.0) + eps) # the projection is not feasible
            @inbounds for i = 1:n
                mu[i] += t_min * (lambda[i] - mu[i])
            end
            f_add = true
            i_add = i_min
            nak += 1
            ak[nak] = i_min
            k = 0
            @inbounds for i = 1:npk
                if pk[i] == i_min
                    k = i
                    break
                end
            end
            if k == 0  # TODO DEBUGGING
                error("_qp1: Cannot find 'i_min' index in 'pk'.")
            end
            kp1 = k + 1
            @inbounds for i = kp1:npk
                pk[i-1] = pk[i]
            end
            pk[npk] = 0
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
                  f_del = true
                  i_del = ii
                  npk += 1
                  pk[npk] = ii
                  ip1 = i + 1
                  @inbounds for k = ip1:nak
                      ak[k-1] = ak[k]
                  end
                  ak[nak] = 0
                  nak -= 1
                  break     #.. for i = 1:nak
            end #.. for i = 1:nak

            if f_opt == true
                break #..Main cycle
            end

        end #.. t_min < (T(1.0) + eps)

    end #..Main cycle

    active = zeros(Int, m2)
    k = 0
    @inbounds for i = 1:nak
        ii = ak[i]
        if ii <= m1
            continue
        end
        k += 1
        active[k] = ii - m1
        if active[k] > m2
            active[k] -= m2
            active[k] = -active[k]
        end
    end

    spl = NormalSpline(spline._kernel,
                  spline._compression,
                  spline._nodes,
                  spline._nodes_b,
                  spline._values,
                  spline._values_lb,
                  spline._values_ub,
                  spline._d_nodes,
                  spline._d_nodes_b,
                  spline._es,
                  spline._es_b,
                  spline._d_values,
                  spline._d_values_lb,
                  spline._d_values_ub,
                  spline._min_bound,
                  cleanup ? nothing : spline._gram,
                  cleanup ? nothing : spline._chol,
                  mu,
                  active,
                  spline._cond,
                  f_opt ? -nit_done : nit_done
                 )

    if logging == true
        open(path,"a") do io
             println(io,"\r\n_qp1 completed.")
        end
    end
    return spl
end
