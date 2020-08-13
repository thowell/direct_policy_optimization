include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
include("../src/velocity.jl")
using Plots

# Horizon
T = 20
model.Δt = 0.05

# Initial and final states
x1 = [0.0; 0.0; 1.0]
xT = [1.0; 0.0; 0.0]

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

model.α = 100.0
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

x_nom = [X_nom[t][1] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)
plot(x_nom)
plot(z_nom)

# Simulate

# simulation time step

function simulate(model,xpp,xp,dt_sim,tf;
		tol=1.0e-6,c_tol=1.0e-6,α=100.0)

    T_sim = floor(convert(Int,tf/dt_sim)) + 1

	# Bounds
	# xl <= x <= xu
	xu_sim = [xpp,xp,Inf*ones(model.nx)]
	xl_sim = [xpp,xp,-Inf*ones(model.nx)]

	# ul <= u <= uu
	uu_sim = Inf*ones(model.nu)
	uu_sim[model.idx_u] .= 0.0
	ul_sim = zeros(model.nu)
	ul_sim[model.idx_u] .= 0.0

	# h = h0 (fixed timestep)
	hu_sim = dt_sim
	hl_sim = dt_sim

	model.α = α
	penalty_obj = PenaltyObjective(model.α)
	multi_obj = MultiObjective([penalty_obj])

	X_traj = [xpp,xp]
	U_traj = []

	for t = 1:T_sim
		# xl <= x <= xu
		xu_sim = [X_traj[t],X_traj[t+1],Inf*ones(model.nx)]
		xl_sim = [X_traj[t],X_traj[t+1],-Inf*ones(model.nx)]

		# Problem
		prob_sim = init_problem(model.nx,model.nu,3,model,multi_obj,
		                    xl=xl_sim,
		                    xu=xu_sim,
		                    ul=[ul_sim],
		                    uu=[uu_sim],
		                    hl=[dt_sim],
		                    hu=[dt_sim]
		                    )
		# MathOptInterface problem
		prob_sim_moi = init_MOI_Problem(prob_sim)

		# Pack trajectories into vector
		Z0_sim = pack([X_traj[t],X_traj[t+1],X_traj[t+1]],[t == 1 ? rand(model.nu) : U_traj[t-1]],dt_sim,prob_sim)

		@time Z_sim_sol = solve(prob_sim_moi,copy(Z0_sim),tol=tol,c_tol=c_tol)
		X_sol, U_sol, H_sol = unpack(Z_sim_sol,prob_sim)

		push!(X_traj,X_sol[end])
		push!(U_traj,U_sol[1])
	end
	return X_traj, U_traj
end

X_sim, U_sim = simulate(model,X_nom[1],X_nom[2],0.01,1.0)

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
model.Δt = 0.01
visualize!(vis,model,[X_sim])
