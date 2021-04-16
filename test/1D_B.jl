@testset "Test 1D_B" begin
    x = [0.0, 1.0, 2.0]                        # function nodes
    x_b = [0.5, 1.5, 2.5]                      # function nodes
    u = x.^2                                   # function values in nodes
    u_lb = u .- 0.1
    u_ub = u .+ 0.1
    s = [2.0]                                  # function first derivative nodes
    v = [4.0]                                  # function first derivative values

    @testset "Test 1D_B_RK_H0 kernel" begin
        spl = smooth(x, u, x_b, u_lb, u_ub, 100, RK_H0(0.1))  # create spline
        cond = estimate_cond(spl)                 # get estimation of the gram matrix condition number
        @test cond ≈ 1000.0
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

    # @testset "Test 1D_B_H1 kernel" begin
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
    # end
    # #
    # @testset "Test 1D_B_H2 kernel" begin
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
    # end
end
