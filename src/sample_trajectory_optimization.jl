using LinearAlgebra, ForwardDiff, StaticArrays
using MathOptInterface, Ipopt
const MOI = MathOptInterface

include("utils.jl")
include("integration.jl")
include("indices.jl")

include("policy.jl")

include("problem.jl")
include("sample_problem.jl")
#
include("objective.jl")
include("sample_objective.jl")
include("sample_general_objective.jl")

#
include("dynamics_constraints.jl")
include("contact_dynamics_constraints.jl")

include("stage_constraints.jl")
include("general_constraints.jl")

#
include("sample_dynamics_constraints.jl")
include("sample_contact_dynamics_constraints.jl")

include("sample_policy_constraints.jl")
#
include("sample_stage_constraints.jl")
include("sample_disturbances.jl")

include("moi.jl")
include("simulate.jl")
