using NormalSmoothingSplines
using DoubleFloats
using Test

@testset "NormalSmoothingSplines.jl" begin

include("1d.jl")
include("2d.jl")
include("3d.jl")

end
