@testset "Test 1D_B" begin
    # x = [0.0, 1.0, 2.0]                        # function nodes
    # x_b = [0.5, 1.5, 2.5]                      # function nodes
    # u = x^2
    # u_b = x_b^2                                    # function values in nodes
    # u_lb = u_b - 1.
    # u_ub = u_b + 1.
    # s = [2.0]                                  # function first derivative nodes
    # v = [4.0]                                  # function first derivative values

    @testset "Test 1D_B_RK_H0 kernel" begin
        x = [4.]
        x_b = [1., 2., 3.]
        u = [10.0]
        u_ub = [ 1., 3., 3.]
        u_lb = [ 0., 2., 0.]
        points = [0., 1., 2., 3., 4.]

        spl = approximate(x, u, x_b, u_lb, u_ub, 100, RK_H0(0.1))
        res = evaluate(spl, points)
        res_rounded = round.(res; digits=3)
        @test res_rounded ≈ [0.967, 1.0, 2.0, 3.0, 10.0]
        @test issetequal(spl._active, [1, 3, -2])
        cond = estimate_cond(spl)
        @test cond ≈ 100.0
#####
        x = [0., 5., 10.]
        x_b = [1., 2., 3., 4., 6., 7., 8., 9.]
        u = [0.0, 10.0, 0.]
        u_ub = [0.1, 1., 1., 10., 10., 1., 1., 0.1]
        u_lb = [0., 0., 0., 0., 0., 0., 0., 0.]
        points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

        spl = approximate(x, u, x_b, u_lb, u_ub, 100, RK_H0(0.1))
        res = evaluate(spl, points)
        res_rounded = round.(res; digits=3)
        @test res_rounded ≈ [0.0, 0.1, 0.55, 1.0, 5.5, 10.0, 5.5, 1.0, 0.55, 0.1, -0.0]
        @test issetequal(spl._active, [8, 1, 6, 3, 0, 0, 0, 0])
        cond = estimate_cond(spl)
        @test cond ≈ 10000.0
##
        spl_inter = interpolate(x, u, RK_H0(0.1))
        res_inter = evaluate(spl_inter, points)
        res_inter_rounded = round.(res_inter; digits=3)
        @test res_inter_rounded ≈ [0.0, 1.999, 3.999, 5.998, 7.999, 10.0, 7.999, 5.998, 3.999, 1.999, 0.0]
        cond_inter = estimate_cond(spl_inter)
        @test cond_inter ≈ 100.0
####
        # σ = evaluate(spl, x)                # evaluate spline in nodes
        # @test σ ≈ u                         # compare with exact function values in nodes
        #
        # spl = prepare(x, RK_H0(0.1))
        # spl = construct(spl, u)
        # σ = evaluate(spl, x)
        # @test σ ≈ u
        # #
        # # Check that we get close when evaluating near the nodes
        # p = x .+ 1e-4*randn(size(x))   # evaluation points near the nodes
        # f = p.^2                       # exact function values in evaluation points
        # σ = evaluate(spl, p)           # evaluate spline in evaluation points
        # # compare spline values with exact function values in evaluation point
        # @test all(isapprox.(σ, f, atol = 0.05))
        #
        # σ = evaluate_one(spl, p[3])
        # @test σ ≈ f[3] atol = 0.05
    end

     @testset "Test 1D_B_H1 kernel" begin
         x = [0., 5., 10.]
         x_b = [1., 2., 3., 4., 6., 7., 8., 9.]
         u = [0.0, 10.0, 0.]
         u_ub = [0.1, 1., 1., 10., 10., 1., 1., 0.1]
         u_lb = [0., 0., 0., 0., 0., 0., 0., 0.]
         points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

         spl = approximate(x, u, x_b, u_lb, u_ub, 100, RK_H1(0.1))
         res = evaluate(spl, points)
         res_rounded = round.(res; digits=3)
         @test res_rounded ≈ [0.0, 0.039, 0.0, 1.0, 6.321, 10.0, 6.321, 1.0, 0.0, 0.039, 0.0]
         @test issetequal(spl._active, [6, 3, -2, -7, 0, 0, 0, 0])
         cond = estimate_cond(spl)
         @test cond ≈ 1.0e7

         spl_inter = interpolate(x, u, RK_H1(0.1))
         res_inter = evaluate(spl_inter, points)
         res_inter_rounded = round.(res_inter; digits=3)
         @test res_inter_rounded ≈ [0.0, 2.913, 5.627, 7.885, 9.428, 10.0, 9.428, 7.885, 5.627, 2.913, -0.0]
         cond_inter = estimate_cond(spl_inter)
         @test cond_inter ≈ 100000.0

###
         x = [0., 5., 10.]
         x_b = [1., 2., 3., 4., 6., 7., 8., 9.]
         u = [0.0, 10.0, 0.]
         u_ub = [Inf, 1., Inf, 10., 10., 1., 1., 0.1]
         u_lb = [0., 0., 0., 0., 0., 0., 0., -Inf]
         points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

         spl = approximate(x, u, x_b, u_lb, u_ub, 100, RK_H1(0.1))
         res = evaluate(spl, points)
         res_rounded = round.(res; digits=3)
         @test res_rounded ≈ [-0.0, 0.0, 1.0, 4.821, 9.028, 10.0, 5.77, 1.0, -0.0, -0.009, 0.0]
         @test issetequal(spl._active, [6, 2, -7, -1, 0, 0, 0, 0])
         cond = estimate_cond(spl)
         @test cond ≈ 1.0e7

###
        x_b = [1., 2., 3., 4., 6., 7., 8., 9.]
        u_ub = [Inf, 1., Inf, 10., 10., 1., 1., 0.1]
        u_lb = [0., 0., 10., 0., 0., 0., 0., -Inf]
        points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

        spl = approximate(x_b, u_lb, u_ub, 100, RK_H1(0.1))
        res = evaluate(spl, points)
        res_rounded = round.(res; digits=3)
        @test res_rounded ≈[1.691, -0.0, 1.0, 10.0, 10.0, 6.622, 3.343, 1.0, 0.0, -0.69, -1.362]
        @test issetequal(spl._active, [-3, 2, 6, -7, -1, 4, 0, 0])
        cond = estimate_cond(spl)
        @test cond ≈ 1.0e6

    #     spl = interpolate(x, u, RK_H1(0.1))  # create spline
    #     cond = estimate_cond(spl)              # get estimation of the gram matrix condition number
    #     @test cond ≈ 1.0e5
    #     σ = evaluate(spl, x)           # evaluate spline in nodes
    #     @test σ ≈ u                    # compare with exact function values in nodes
    #
    #     d1_σ = evaluate_derivative(spl, 1.0)     # evaluate spline first derivative in the node
    #     @test d1_σ ≈ 2.0 atol = 0.05              # compare with exact function first derivative value in the node
    #
    #     # Check that we get close when evaluating near the nodes
    #     p = x .+ 1e-4*randn(size(x))   # evaluation points near the nodes
    #     f = p.^2                       # exact function values in evaluation points
    #     σ = evaluate(spl, p)                # evaluate spline in evaluation points
    #     # compare spline values with exact function values in evaluation point
    #     @test all(isapprox.(σ, f, atol = 1e-2))
    #
    #     ###
    #     spl = interpolate(x, u, s, v, RK_H1(0.1)) # create spline by function and
    #                                         # first derivative values in nodes
    #     cond = estimate_cond(spl)              # get estimation of the gram matrix condition number
    #     @test cond ≈ 1.0e5
    #     σ = evaluate(spl, x)                # evaluate spline in nodes
    #     @test σ ≈ u                    # compare with exact function values in nodes
    #
    #     # Check that we get close when evaluating near the nodes
    #     p = x .+ 1e-3*randn(size(x))   # evaluation points near the nodes
    #     f = p.^2                       # exact function values in evaluation points
    #     σ = evaluate(spl, p)                # evaluate spline in evaluation points
    #     # compare spline values with exact function values in evaluation point
    #     @test all(isapprox.(σ, f, atol = 1e-2))
    #
    #     spl = prepare(x, s, RK_H1(0.1))
    #     spl = construct(spl, u, v)
    #     σ = evaluate_one(spl, p[3])
    #     @test σ ≈ f[3] atol = 0.05
     end
    # #
    @testset "Test 1D_B_H2 kernel" begin
        x = [4.]
        x_b = [1., 2., 3.]
        u = [10.0]
        u_ub = [ 1., 3., 3.]
        u_lb = [ 0., 2., 0.]
        points = [0., 1., 2., 3., 4.]

        spl = approximate(x, u, x_b, u_lb, u_ub, 100, RK_H2(0.1))
        res = evaluate(spl, points)
        res_rounded = round.(res; digits=3)
        @test res_rounded ≈ [-2.263, 1.0, 2.0, 3.0, 10.0]
        @test issetequal(spl._active, [1, 3, -2])
        cond = estimate_cond(spl)
        @test cond ≈ 1.0e7
##
        x = [0., 5., 10.]
        x_b = [1., 2., 3., 4., 6., 7., 8., 9.]
        u = [0.0, 10.0, 0.]
        u_ub = [0.1, 1., 1., 10., 10., 1., 1., 0.1]
        u_lb = [0., 0., 0., 0., 0., 0., 0., 0.]
        points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

        spl = approximate(x, u, x_b, u_lb, u_ub, 100, RK_H2(0.1))
        res = evaluate(spl, points)
        res_rounded = round.(res; digits=3)
        @test res_rounded ≈ [-0.0, 0.1, -0.0, 1.0, 6.298, 10.0, 6.298, 1.0, -0.0, 0.1, 0.0]
        @test issetequal(spl._active, [1, 8, 6, 3, -2, -7, 0, 0])
        cond = estimate_cond(spl)
        @test cond ≈ 1.0e12
##
        x_b = [1., 2., 3., 4., 6., 7., 8., 9.]
        u_ub = [Inf, 1., Inf, 10., 10., 1., 1., 0.1]
        u_lb = [0., 0., 10., 0., 0., 0., 0., -Inf]
        points = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

        spl = approximate(x_b, u_lb, u_ub, 100, RK_H2(0.1))
        res = evaluate(spl, points)
        res_rounded = round.(res; digits=2)
        @test res_rounded ≈ [14.75, 0.0, 1.0, 10.0, 10.0, 4.45, 0.83, 0.0, 0.14, 0.1, -0.35]
        @test issetequal(spl._active, [-3, 2, -1, 4, 8, -6, 0, 0])
        cond = estimate_cond(spl)
        @test cond ≈ 1.0e10

    #     spl = interpolate(x, u, RK_H2(0.1))  # create spline
    #     cond = estimate_cond(spl)              # get estimation of the gram matrix condition number
    #     @test cond ≈ 1.0e7
    #     σ = evaluate(spl, x)           # evaluate spline in nodes
    #     @test σ ≈ u                    # compare with exact function values in nodes
    #
    #     d1_σ = evaluate_derivative(spl, 1.0)     # evaluate spline first derivative in the node
    #     @test d1_σ ≈ 2.0 atol = 0.005              # compare with exact function first derivative value in the node
    #
    #     # Check that we get close when evaluating near the nodes
    #     p = x .+ 1e-4*randn(size(x))   # evaluation points near the nodes
    #     f = p.^2                       # exact function values in evaluation points
    #     σ = evaluate(spl, p)                # evaluate spline in evaluation points
    #     # compare spline values with exact function values in evaluation point
    #     @test all(isapprox.(σ, f, atol = 1e-2))
    #
    #     ###
    #     spl = interpolate(x, u, s, v, RK_H2(0.1)) # create spline by function and
    #                                               # first derivative values in nodes
    #     σ = evaluate(spl, x)                # evaluate spline in nodes
    #     @test σ ≈ u                    # compare with exact function values in nodes
    #
    #     # Check that we get close when evaluating near the nodes
    #     p = x .+ 1e-3*randn(size(x))   # evaluation points near the nodes
    #     f = p.^2                       # exact function values in evaluation points
    #     σ = evaluate(spl, p)                # evaluate spline in evaluation points
    #     # compare spline values with exact function values in evaluation point
    #     @test all(isapprox.(σ, f, atol = 1e-2))
    end
end
