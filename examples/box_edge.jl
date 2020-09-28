include("../src/sample_trajectory_optimization.jl")
include("../dynamics/box.jl")
using Plots

# Horizon
tf = 2.0
T = 20
Δt = tf/T

# Initial and final states
_mrp_corner = MRP(UnitQuaternion(RotY(-1.0*atan(1/sqrt(2.0)))*RotX(pi/4)))
_mrp_edge = MRP(UnitQuaternion(RotY(pi/2)*RotX(pi/2)))
_mrp = MRP(UnitQuaternion(RotY(0)*RotX(0)))

x1 = [model.r; model.r; model.r;_mrp.x;_mrp.y;_mrp.z]
xT = [2*model.r; -1.0*model.r; model.r; _mrp_edge.x;_mrp_edge.y;_mrp_edge.z]
# xT = [0.0; 0.0; model.r; _mrp_corner.x;_mrp_corner.y;_mrp_corner.z]
x_ref = [x1, linear_interp(x1,xT,T-1)...]

# x_ref = [x1, linear_interp(x1,xM,11)[1:end-1]..., linear_interp(xM,xT,9)...]
# vis = Visualizer()
# open(vis)

visualize_box!(vis,model,x_ref,Δt=Δt)

# Bounds
# xl <= x <= xu
xu_traj = [Inf*ones(model.nx) for t=1:T]
xl_traj = [-Inf*ones(model.nx) for t=1:T]

xu_traj[1] = x1
xl_traj[1] = x1

xu_traj[2] = x1
xl_traj[2] = x1

# xu_traj[T] = xT
# xl_traj[T] = xT


# ul <= u <= uu
uu_traj = [[Inf*ones(model.nu_ctrl);Inf*ones(model.nu-model.nu_ctrl)] for t = 1:T-2]
ul_traj = [[-Inf*ones(model.nu_ctrl);zeros(model.nu-model.nu_ctrl)] for t = 1:T-2]

# uu_traj[1][1:3] .= 0.0
# ul_traj[1][1:3] .= 0.0

# h = h0 (fixed timestep)
hu = Δt
hl = Δt

# Objective
Q = [(t<T ? Diagonal(1.0*ones(model.nx))
	: Diagonal(100.0*ones(model.nx))) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu_ctrl)) for t = 1:T-2]
c = 0.0

obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])

model.α = 100.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = deepcopy(x_ref)# linear interpolation on state #TODO clip z
U0 = [1.0e-5*rand(model.nu) for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0),c_tol=1.0e-2,tol=1.0e-2)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

x_nom = [X_nom[t][1] for t = 1:T]
y_nom = [X_nom[t][2] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]
u_nom = [U_nom[t][model.idx_u] for t = 1:T-2]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
b_nom = [U_nom[t][model.idx_b] for t = 1:T-2]
ψ_nom = [U_nom[t][model.idx_ψ[1]] for t = 1:T-2]
η_nom = [U_nom[t][model.idx_η] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)

plot(hcat(u_nom...)',linetype=:steppost)

using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshIO
using Rotations
using Meshing

vis = Visualizer()
open(vis)

visualize!(vis,model,X_nom,Δt=H_nom[1])
