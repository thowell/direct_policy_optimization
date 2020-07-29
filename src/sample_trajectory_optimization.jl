using LinearAlgebra, ForwardDiff, StaticArrays
using MathOptInterface, Ipopt
const MOI = MathOptInterface

include("utils.jl")
include("integration.jl")
include("indices.jl")
include("objective.jl")
include("dynamics_constraints.jl")
include("stage_constraints.jl")
include("problem.jl")
include("sample_problem.jl")
include("moi.jl")
include("simulate.jl")
