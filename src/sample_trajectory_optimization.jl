using LinearAlgebra, ForwardDiff, StaticArrays
using MathOptInterface, Ipopt
const MOI = MathOptInterface

include("utils.jl")
include("integration.jl")
include("indices.jl")

include("problem.jl")
include("sample_problem.jl")

include("objective.jl")
include("sample_objective.jl")

include("dynamics_constraints.jl")
include("stage_constraints.jl")
include("general_constraints.jl")

include("sample_dynamics_constraints.jl")
include("sample_control_constraints.jl")
include("sample_stage_constraints.jl")
include("sample_disturbances.jl")

include("moi.jl")
include("simulate.jl")
