include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
using Plots

# Horizon
T = 10

# Initial and final states
x1 = [0.0; 0.0; 1.0]
xT = [0.0; 0.0; 0.0]

# Bounds
# xl <= x <= xu
xu_traj = [Inf*ones(model.nx) for t=1:T]
xl_traj = [-Inf*ones(model.nx) for t=1:T]

# xu_traj[1] = x1
xu_traj[2] = x1

# xl_traj[1] = x1
xl_traj[2] = x1

xu_traj[T] = xT
xl_traj[T] = xT

# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 10.0
ul = zeros(model.nu)
ul[model.idx_u] .= -10.0

# h = h0 (fixed timestep)
tf0 = 1.0
h0 = tf0/(T-1)
hu = h0
hl = h0

# Objective
Q = [t < T ? Diagonal(ones(model.nx)) : Diagonal(10.0*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu_ctrl)) for t = 1:T-2]
c = 0.0

obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])
penalty_obj = PenaltyObjective(10.0)
multi_obj = MultiObjective([obj,penalty_obj])

function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]] - [1.0;0.0;0.0]
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	nx = prob.nx
	idx = prob.idx

	shift = 0
	# c[1:nx] = (Z[idx.x[2]] - Z[idx.x[1]])/Z[idx.h[1]]

	r_idx = 1:nx

	c_idx = idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(-1.0/Z[idx.h[1]]*ones(nx)))
	shift += len

	c_idx = idx.x[2]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(Diagonal(1.0/Z[idx.h[1]]*ones(nx)))
	shift += len

	c_idx = idx.h[1]
	len = length(r_idx)*length(c_idx)
	∇c[shift .+ (1:len)] = vec(-1.0*(Z[idx.x[2]] - Z[idx.x[1]])/(Z[idx.h[1]]*Z[idx.h[1]]))
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

	r_idx = r_shift .+ (1:nx)

	c_idx = idx.x[1]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.x[2]
	row_col!(row,col,r_idx,c_idx)

	c_idx = idx.h[1]
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
					m_general=model.nx,
					general_ineq=(1:0)
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state #TODO clip z
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

x = [X_nom[t][1] for t = 1:T]
z = [X_nom[t][3] for t = 1:T]
λ = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
s = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s)
plot(x)
plot(z)
