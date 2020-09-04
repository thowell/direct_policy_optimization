include("../src/direct_policy_optimization.jl")
include("../dynamics/biped.jl")
using Plots
using TrajectoryOptimization
using Altro
using RobotDynamics
using StaticArrays
using LinearAlgebra
const TO = TrajectoryOptimization
const RD = RobotDynamics

struct BIPED{T} <: TO.AbstractModel
    x::T
end

function RD.dynamics(model::BIPED, x, u)
    f(x) + g(x)*u
end

Base.size(::BIPED) = 10,4

# Model and discretization
model_altro = BIPED(1.0)
n, m = 10,4
tf = 0.36915  # sec
N = 10   # number of knot points

# Objective
x0 = SA[2.44917526290273,2.78819438807838,0.812693850088907,0.951793806080012,1.49974183648719e-13,0.0328917694318260,0.656277832705193,0.0441573173000750,1.03766701983449,1.39626340159558]  # initial state
xf = SA[2.84353387648829,2.48652580597252,0.751072212241267,0.645830978766432, 0.0612754113212848,2.47750069399969,3.99102008940145,-1.91724136219709,-3.95094757056324,0.0492401787458546]  # final state

Q = Diagonal(@SVector [0.0,0.0,0.0,0.0,0.0,1.0e-4,1.0e-4,1.0e-4,1.0e-4,1.0e-4])
R = Diagonal(1.0e-3*@SVector ones(m))
obj = LQRObjective(Q, R, Q, xf, N)

# Constraints
cons = ConstraintList(n,m,N)
add_constraint!(cons, GoalConstraint(xf), N:N)
add_constraint!(cons, BoundConstraint(n,m, u_min=-20.0, u_max=20.0), 1:N-1)


# Create and solve problem
prob = TO.Problem(model_altro, obj, xf, tf, x0=x0, constraints=cons)
solver = ALTROSolver(prob)
solver.opts.projected_newton=false
solver.opts.constraint_tolerance = 1.0e-3
cost(solver)           # initial cost
@time Altro.solve!(solver)   # solve with ALTRO
max_violation(solver)  # max constraint violation
cost(solver)           # final cost
iterations(solver)     # total number of iterations

# Get the state and control trajectories
X = states(solver)
U = controls(solver)

T = N
Q_nominal = [X[t][1:5] for t = 1:T]
foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=1:T]
foot_y = [foot_traj[t][2] for t=1:T]

plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)
