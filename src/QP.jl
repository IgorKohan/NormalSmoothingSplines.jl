function _qp(spline::NormalSpline{T, RK}, nit::Int, tol::T,
             cleanup::Bool = false
            ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    logging::Bool = true # DEBUGGING
    if logging
        path = "0_qp.log"
        if ispath(path)
           rm(path)
        end
        open(path,"a") do io
            println(io,"_qp started.\r\n")
        end
    end

# Initialization..
    m1 = 0
    if !isnothing(spline._nodes)
        m1 = size(spline._nodes, 2)
    end
    m2 = size(spline._nodes_b, 2)

    if m1 > 0
        b = [spline._values; spline._values_ub; -spline._values_lb]
    else
        b = [spline._values_ub; -spline._values_lb]
    end

    m1p1 = m1 + 1
    m2pm2 = m2 + m2
    m = m1 + m2
    n = m1 + m2 + m2

    nak = m1
    ak = zeros(Int, m)
    @inbounds for j = 1:nak
        ak[j] = j
    end

    pk = zeros(Int, m2pm2)
    npk = m2pm2
    @inbounds for j = 1:m2pm2
        pk[j] = j + m1
    end

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
    i_del = 0

# Main cycle
    mat = nothing
    nit_done = 0
    nit_fac = 0
#
    @inbounds for it = 1:nit
        nit_done += 1
        nit_fac += 1

        if logging
            open(path,"a") do io
                println(io,"\r\nIteration:", it)
                if it > 1 && f_add
                    println(io,"\r\nConstraint added.")
                end
                if it > 1 && f_del
                    println(io,"\r\nConstraint released.")
                end
                nab = length(ak[ak .> 0]) - m1
                nlb = length(ak[ak .> m])
                println(io,"\r\nActive constraints:", (nak-m1))
                println(io,"\r\nActive upper constraints:", (nab-nlb))
                println(io,"\r\nActive lower constraints:", nlb)
                println(io,"\r\nInactive constraints:", npk)
            end
        end

        @inbounds for i = 1:n
            lambda[i] = T(0.)
        end

        if nak > 0
#  Calculating Gram matrix factorization and lambda
            if it == 1 || nak <= 2 # TODO change it
                f_add = false
            end
            if it == 1 || (i_del > 0 && i_del <= 1) # TODO change it
                f_del = false
            end

            if nit_fac > n / 2
                f_add = false
                f_del = false
                nit_fac = 0
            end

            w = zeros(T, nak)
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
                end # for j = 1:nak

                try
                    mat = cholesky!(mat)
                catch
                    if logging
                        open(path,"a") do io
                             println(io,"\r\n_qp: Gram matrix 'mat' is degenerate.\r\n")
                        end # io
                    end # logging
                    error("_qp: Gram matrix 'mat' is degenerate.")
                end

                #  Calculating lambda
                w = mat \ w

                if logging
                    open(path,"a") do io
                        println(io,"\r\nFactorization (full) is calculated.")
                    end
                end
            end #if !f_add && !f_del

            if f_del
                chol = Matrix{T}(undef, nak, nak)
                nakp1 = nak + 1

                if i_del == nakp1
                    @inbounds for j = 1:nak
                          for i = j:nak
                              chol[j,i] = mat.U[j,i]
                              chol[i,j] = chol[j,i]
                          end
                    end
                    mat = Cholesky(chol, :U, 0)
                else
                    rec_err::Bool = false
                    te = nakp1 - i_del
                    for t = 1:te
                        l = i_del + t
                        g1::T = mat.L[l, l-1]
                        g2::T = mat.L[l, l]
                        hp::T = hypot(g1, g2)
                        if hp < eps(T)
                            if logging
                                open(path,"a") do io
                                     println(io,"\r\n_qp: updated (-) Gram matrix is degenerate.\r\n")
                                     println(io,"_qp: Going to full Gram matrix factorization.\r\n")
                                end # io
                            end # logging

                            rec_err = true
                            break # for t = 1:te
                        end
                        c::T = g1 / hp;
                        s::T = g2 / hp;

                        for k = l:nakp1
                            g1 = mat.L[k, l-1]
                            g2 = mat.L[k, l]
                            a1 =  c * g1 + s * g2
                            a2 = -s * g1 + c * g2
                            mat.factors[k, l-1] = c * g1 + s * g2
                            mat.factors[k, l] = -s * g1 + c * g2
                            mat.factors[l-1, k] = mat.factors[k, l-1]
                            mat.factors[l, k] = mat.factors[k, l]
                        end
                    end # for t = 1:te
                    if rec_err
                        break # Main cycle
                    end

                    k = 0
                    @inbounds for i = 1:nak
                        k += 1
                        if i == i_del
                             k += 1
                        end
                        for j = 1:i
                            chol[i,j] = mat.L[k,j]
                        end
                    end

                    mat = Cholesky(chol, :L, 0)
                end # if i_del == nakp1

                #  Calculating lambda
                w = mat \ w

                if logging
                    open(path,"a") do io
                        println(io,"\r\nFactorization is recalculated (-).")
                    end
                end
            end # if f_del

            if f_add
                nakm1 = nak - 1
                gram_col = zeros(T, nak-1)
                gram_el = T(0.)
                si = T(1.)
                sj = T(1.)
                ii = ak[nak]
                if ii > m
                    ii -= m2
                    si = T(-1.)
                end
                @inbounds for j = 1:nak
                      jj = ak[j]
                      if jj > m
                          jj -= m2
                          sj = T(-1.)
                      else
                          sj = T(1.)
                      end
                      if j < nak
                          gram_col[j] = si * sj * spline._gram[ii, jj]
                      else
                          gram_el = si * sj * spline._gram[ii, jj]
                      end
                end # for j = 1:nak

                chol = Matrix{T}(undef, nak, nak)
                @inbounds for j = 1:nakm1
                      for i = j:nakm1
                          chol[j,i] = mat.U[j,i]
                          chol[i,j] = chol[j,i]
                      end
                end
                gram_col = mat.L \ gram_col
                for i = 1:nakm1
                    chol[nak,i] = gram_col[i]
                    chol[i,nak] = gram_col[i]
                end
                chol_el = gram_el - gram_col' * gram_col
                if chol_el > T(0.)
                    chol[nak,nak] = sqrt(chol_el)
                else
                    if logging
                        open(path,"a") do io
                             println(io,"\r\n_qp: updated (+) Gram matrix is degenerate.\r\n")
                             println(io,"_qp: Going to full Gram matrix factorization.\r\n")
                        end # io
                    end # logging

                    # Going to full Gram matrix factorization.
                    f_add = false
                    break # Main cycle
                end

                #  Calculating lambda
                mat = Cholesky(chol, :U, 0)
                w = mat \ w

                if logging
                    open(path,"a") do io
                        println(io,"\r\nFactorization is recalculated. (+)")
                    end
                end
            end #if f_add

            @inbounds for j = 1:nak
                  jj = ak[j]
                  lambda[jj] = w[j]
            end
            #..Calculating lambda

        end #if nak > 0
#..Calculating Gram matrix factorization and lambda

        if logging
            open(path,"a") do io
                 norm = T(0.)
                 @inbounds for j = 1:m1
                     for i = 1:m1
                         norm += spline._gram[j,i]*mu[j]*mu[i]
                     end
                 end
                 @inbounds for j = m1p1:m
                     for i = m1p1:m
                         norm += spline._gram[j,i]*(mu[j] - mu[j+m2])*(mu[i] - mu[i+m2])
                     end
                 end
                 @inbounds for i = 1:m1
                     for j = m1p1:m
                         norm += T(2.) * spline._gram[i,j]*mu[i]*(mu[j] - mu[j+m2])
                     end
                 end
                 norm = sqrt(norm)
                 norm = round(norm; digits=0)
                 println(io,"\r\nSpline norm:", norm)

                 # println(io,"\r\nNumber of active inequality constraints:", nak-m1)
                 # println(io,"\r\nActive inequality constraints:\r\n")
                 # l_ak = copy(ak)
                 # for i= m1p1:nak
                 #     if l_ak[i] > m
                 #        l_ak[i] -= m
                 #        l_ak[i] = -l_ak[i]
                 #     else
                 #        l_ak[i] -= m1
                 #     end
                 # end
                 # if nak > m1
                 #     println(io, l_ak[m1p1:end])
                 # end
                 # println(io,"\r\nNumber of inactive inequality constraints:", npk)
                 # println(io,"\r\nInactive inequality constraints:\r\n")
                 # l_pk = copy(pk)
                 # for i= 1:npk
                 #     if l_pk[i] > m
                 #        l_pk[i] -= m
                 #        l_pk[i] = -l_pk[i]
                 #     else
                 #        l_pk[i] -= m1
                 #     end
                 # end
                 # println(io, l_pk)
                 println(io, "\r\n")
            end # io
        end # logging

        if nak == m
            @inbounds for i = 1:n
                mu[i] = lambda[i]
            end

            f_opt = true
            @inbounds for i = 1:nak
                  ii = ak[i]
                  if ii <= m1
                      continue #.. for i = 1:nak
                  end
                  if lambda[ii] < T(tol)
                      continue #.. for i = 1:nak
                  end
                  f_opt = false
                  f_del = true
                  f_add = false
                  i_del = i
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
        si = T(1.)
        sj = T(1.)
        t_min::T = t_max
        i_min = 0
        @inbounds for i = 1:npk
            eik = T(0.)
            s = T(0.)
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

            if eik > T(tol)
                ii = pk[i]
                if abs(b[ii]) != Inf
                    tik = (b[ii] - s) / eik  # tik >≈ 0 here
                    if tik < t_min
                        i_min = ii
                        t_min = tik
                    end
                end
            end
        end #.. for i = 1:npk
#.. Calculating t_min, i_min
        if t_min < T(tol)  # fixing the round-off error
            t_min = T(0.)
        end
        if t_min >= T(0.) && t_min < (T(1.) + tol) # the projection is not feasible
            @inbounds for i = 1:n
                mu[i] += t_min * (lambda[i] - mu[i])
            end
            f_add = true
            f_del = false
            i_del = 0
            nak += 1
            ak[nak] = i_min
            k = 0
            @inbounds for i = 1:npk
                if pk[i] == i_min
                    k = i
                    break
                end
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
                  if lambda[ii] < T(tol)
                      continue #.. for i = 1:nak
                  end
                  f_opt = false
                  f_del = true
                  f_add = false
                  i_del = i
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

        end #.. t_min < (T(1.) + tol)

    end #..Main cycle

    active = zeros(Int, m2)
    k = 0
    @inbounds for i = m1p1:nak
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
                  cleanup ? nothing : mat,
                  mu,
                  active,
                  spline._cond,
                  f_opt ? -nit_done : nit_done
                 )

    if logging
        open(path,"a") do io
             if(f_opt)
                 println(io,"\r\nSolution found.\r\n")
             else
                 println(io,"\r\nLimit of iterations.\r\n")
             end

             norm = T(0.)
             @inbounds for j = 1:m1
                 for i = 1:m1
                     norm += spline._gram[j,i]*mu[j]*mu[i]
                 end
             end
             @inbounds for j = m1p1:m
                 for i = m1p1:m
                     norm += spline._gram[j,i]*(mu[j] - mu[j+m2])*(mu[i] - mu[i+m2])
                 end
             end
             @inbounds for i = 1:m1
                 for j = m1p1:m
                     norm += T(2.) * spline._gram[i,j]*mu[i]*(mu[j] - mu[j+m2])
                 end
             end
             norm = sqrt(norm)
             norm = round(norm; digits=0)
             nab = length(ak[ak .> 0]) - m1
             nlb = length(ak[ak .> m])
             println(io,"\r\nActive constraints:", (nak-m1))
             println(io,"\r\nActive upper constraints:", (nab-nlb))
             println(io,"\r\nActive lower constraints:", nlb)
             println(io,"\r\nInactive constraints:", npk)
             println(io,"\r\nSpline norm:", norm)

             # println(io,"\r\nNumber of active inequality constraints:", nak-m1)
             # println(io,"\r\nActive inequality constraints:\r\n")
             # l_ak = copy(ak)
             # for i= m1p1:nak
             #     if l_ak[i] > m
             #        l_ak[i] -= m
             #        l_ak[i] = -l_ak[i]
             #     else
             #        l_ak[i] -= m1
             #     end
             # end
             # println(io, l_ak[m1p1:end])
             # println(io,"\r\nNumber of inactive inequality constraints:", npk)
             # println(io,"\r\nInactive inequality constraints:\r\n")
             # l_pk = copy(pk)
             # for i= 1:npk
             #     if l_pk[i] > m
             #        l_pk[i] -= m
             #        l_pk[i] = -l_pk[i]
             #     else
             #        l_pk[i] -= m1
             #     end
             # end
             # println(io, l_pk)
             # println(io, "\r\n")

             println(io,"\r\n_qp completed.\r\n")
        end #io
    end # logging

    return spl
end