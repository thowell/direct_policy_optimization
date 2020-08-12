include("../src/sample_trajectory_optimization.jl")
include("../dynamics/hopper.jl")
using Plots

# Horizon
T = 21
Tm = convert(Int,(T-3)/2 + 3)

tf = 0.5
model.Δt = tf/(T-1)

zh = 0.1
# Initial and final states
x1 = [0., model.r+zh, model.r, 0., 0.]
xM = [0.5, 0.5*model.r, 0.5*model.r, 0., 0.]
xT = [1.0, model.r+zh, model.r, 0., 0.]

# Bounds
# xl <= x <= xu
xu_traj = [model.qU for t=1:T]
xl_traj = [model.qL for t=1:T]

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
qL
# Objective
Q = [t < T ? Diagonal([1.0,1.0,1.0,0.1,0.1]) : Diagonal(5.0*ones(model.nx)) for t = 1:T]
R = [Diagonal([1.0e-1,1.0e-3]) for t = 1:T-2]
c = 1.0

# x1_ref = @SVector [0., model.r+zh, 0.5*model.r, 0., 0.]
# xT_ref = @SVector [1.0, model.r+zh, 0.5*model.r, 0., 0.]
# X_ref = [x1,linear_interp(x1_ref,xM,6)[2:end-1]...,linear_interp(xM,xT_ref,6)[1:end-1]...,xT]
X_ref = linear_interp(x1,xT,T)
X_ref[Tm] = xM

obj = QuadraticTrackingObjective(Q,R,c,
    [X_ref[t] for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])
model.α = 1000.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx
	T = prob.T
	c[1:nx-1] = (Z[idx.x[3]] - Z[idx.x[T]])[2:end]

	v1 = (Z[idx.x[3]] - Z[idx.x[2]])/Z[idx.h[1]]
	vT = (Z[idx.x[T]] - Z[idx.x[T-1]])/Z[idx.h[T-2]]

	c[nx-1 .+ (1:nx)] = v1 - vT
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	shift = 0

	r_idx = 1:nx-1

	c_idx = idx.x[3][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(ones(nx-1)))
	shift += len

	c_idx = idx.x[T][2:end]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx-1)))
	shift += len

	# v1 = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]
	# vT = (Z[idx.x[10]] - Z[idx.x[9]])/Z[idx.h[9]]
	#
	# c[nx-1 .+ (1:nx)] = v1 - vT
	r_idx = nx-1 .+ (1:nx)

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)./Z[idx.h[1]]))
	shift += len

	c_idx = idx.x[3]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(1.0*ones(nx)./Z[idx.h[1]]))
	shift += len

	c_idx = idx.h[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*(Z[idx.x[3]] - Z[idx.x[2]])/(Z[idx.h[1]]*Z[idx.h[1]]))
	shift += len

	c_idx = idx.x[T-1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(1.0*ones(nx)./Z[idx.h[T-2]]))
	shift += len

	c_idx = idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0*ones(nx)./Z[idx.h[T-2]]))
	shift += len

	c_idx = idx.h[9]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(1.0*(Z[idx.x[T]] - Z[idx.x[T-1]])/(Z[idx.h[T-2]]*Z[idx.h[T-2]]))
	shift += len

	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0)

	row = []
	col = []

	nx = prob.nx
	idx = prob.idx

	# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]

	r_idx = r_shift .+ (1:nx-1)

	c_idx = idx.x[3][2:end]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[T][2:end]
	row_col!(row,col,r_idx,c_idx)

	r_idx = r_shift + nx-1 .+ (1:nx)

	c_idx = idx.x[2]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[3]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.h[1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[T-1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[T]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.h[T-2]
	row_col!(row,col,r_idx,c_idx)

	return collect(zip(row,col))
end

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
					general_ineq=(1:0),
					contact_sequence=true,
					T_contact_sequence=[Tm]
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.001*rand(model.nu) for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
plot(hcat(U_nom...)[model.idx_u,:]',linetype=:steppost)
s = [U_nom[t][model.idx_s] for t = 1:T-2]
@assert norm(s,Inf) < 1.0e-5
@assert norm(X_nom[Tm] - xM) < 1.0e-5
@assert norm(X_nom[3][2:end] - X_nom[T][2:end]) < 1.0e-5
λ = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
plot(λ,linetype=:steppost)

using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshIO
using Rotations

vis = Visualizer()
open(vis)
visualize!(vis,model,X_nom)
