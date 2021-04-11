using NormalSmoothingSplines
using DoubleFloats
using Test

@testset "NormalSmoothingSplines.jl" begin

include("1D.jl")
include("2D.jl")
include("3D.jl")

end
