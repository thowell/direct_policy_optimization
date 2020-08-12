include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
include("../src/loop.jl")
using Plots

# Horizon
T = 31
Tm = convert(Int,(T-3)/2 + 3)

tf = 1.0
model.Δt = tf/(T-1)

zh = 0.5
# Initial and final states
x1 = [0.0, 0.0, zh]
xM = [1.0, 0.0, 0.0]
xT = [2.0, 0.0, zh]

# Bounds
# xl <= x <= xu
xu = Inf*ones(model.nx)
xl = -Inf*ones(model.nx)
xu_traj = [xu for t=1:T]
xl_traj = [xl for t=1:T]

xu_traj[3] = x1
xu_traj[Tm] = xM
# xu_traj[T] = xT

xl_traj[3] = x1
xl_traj[Tm] = xM
# xl_traj[T] = xT

# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

# h = h0 (fixed timestep)
hu = 2.0*model.Δt
hl = 0.0*model.Δt

# Objective
Q = [t < T ? Diagonal([1.0,1.0,1.0]) : Diagonal([1.0,1.0,1.0]) for t = 1:T]
R = [Diagonal([1.0e-1,1.0e-1,1.0e-1]) for t = 1:T-2]
c = 1.0

X_ref = linear_interp(x1,xT,T)
X_ref[Tm] = xM

obj = QuadraticTrackingObjective(Q,R,c,
    [X_ref[t] for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])
model.α = 1000.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul for t = 1:T-2],
                    uu=[uu for t = 1:T-2],
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2],
					general_constraints=true,
					m_general=model.nx-1+model.nx,
					general_ineq=(1:0)
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T)
X0[Tm] = xM
U0 = [0.001*rand(model.nu) for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

x_nom = [X_nom[t][1] for t = 1:T]
y_nom = [X_nom[t][2] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]

λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)
plot(x_nom)
plot(y_nom)
plot(z_nom)

plot(hcat(U_nom...)[model.idx_u,:]',linetype=:steppost)
@assert norm(s_nom,Inf) < 1.0e-5
@assert norm(X_nom[Tm] - xM) < 1.0e-5
@assert norm(X_nom[3][2:end] - X_nom[T][2:end]) < 1.0e-5
plot(λ,linetype=:steppost)

# using Colors
# using CoordinateTransformations
# using FileIO
# using GeometryTypes
# using LinearAlgebra
# using MeshCat
# using MeshIO
# using Rotations
#
# vis = Visualizer()
# open(vis)
visualize!(vis,model,X_nom)
