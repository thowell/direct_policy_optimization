using LinearAlgebra, ForwardDiff, FiniteDiff, StaticArrays
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
include("time_constraints.jl")
include("stage_constraints.jl")
include("general_constraints.jl")

include("sample_dynamics_constraints.jl")
# include("sample_time_constraints.jl")
include("sample_control_constraints.jl")
include("sample_state_constraints.jl")
include("sample_stage_constraints.jl")


include("policy.jl")
# include("policy_constraints.jl")


include("moi.jl")
include("simulate.jl")