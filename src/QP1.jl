function qp(spline::NormalSpline{T, RK}, cleanup::Bool
           ) where {T <: AbstractFloat, RK <: ReproducingKernel_0}

    path = "0_qp.log"
    if ispath(path)
       rm(path)
    end
    open(path,"a") do io
        println(io,"qp started.\r\n")
    end

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
                  spline._cond
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
