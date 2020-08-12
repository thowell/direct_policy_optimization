include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
include("../src/velocity.jl")
using Plots

# Horizon
T = 10

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

model.α = 10.0
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
# visualize!(vis,model,X_nom)

# samples
Q_lqr = [t < T ? Diagonal([10.0;10.0;10.0]) : Diagonal([100.0; 100.0; 100.0]) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Samples
N = 2*model.nx
models = [model for i =1:N]
K0 = [rand(model.nu_ctrl,model.nx) for t = 1:T-2]
β = 1.0
w = 1.0e-2*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
	xl_traj_sample[i][2] = x1_sample[i]
	xu_traj_sample[i][2] = x1_sample[i]
end

ul_traj_sample = [[ul for t = 1:T-2] for i = 1:N]
uu_traj_sample = [[uu for t = 1:T-2] for i = 1:N]

hl_traj_sample = [[hl for t = 1:T-2] for i = 1:N]
hu_traj_sample = [[hu for t = 1:T-2] for i = 1:N]

prob_sample = init_sample_problem(prob,models,x1_sample,
    Q_lqr,R_lqr,H_lqr,
	u_policy=model.idx_u,
    β=β,w=w,γ=γ,
    disturbance_ctrl=false,
    α=1.0,
	ul=ul_traj_sample,
	uu=uu_traj_sample,
	xl=xl_traj_sample,
	xu=xu_traj_sample,
	hl=hl_traj_sample,
	hu=hu_traj_sample,
    sample_general_constraints=true,
    m_sample_general=N*model.nx,
    sample_general_ineq=(1:0),
	general_objective=true)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K0,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=1000)
# Z_sample_sol = solve(prob_sample_moi,Z_sample_sol)

X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

s = [U_nom_sample[t][model.idx_s] for t = 1:T-2]
@assert sum(s) < 1.0e-5

pltz = plot(label="",xlabel="time (s)",ylabel="z",title="Particle",
	legend=:topright)
t_span = range(0,stop=model.Δt*(T-1),length=T)
for i = 1:N
	z_sample = [X_sample[i][t][3] for t = 1:T]
	pltz = plot!(t_span,z_sample,label="")
end
pltz = plot!(t_span,z_nom,color=:purple,label="nominal",width=2.0)
z_nom_sample =  [X_nom_sample[t][3] for t = 1:T]
pltz = plot!(t_span,z_nom_sample,color=:orange,label="sample nominal",width=2.0)
display(pltz)

pltx = plot(label="",xlabel="time (s)",ylabel="x",title="Particle",
	legend=:bottomright)
t_span = range(0,stop=model.Δt*(T-1),length=T)
for i = 1:N
	x_sample = [X_sample[i][t][1] for t = 1:T]
	pltx = plot!(t_span,x_sample,label="")
end
pltx = plot!(t_span,x_nom,color=:purple,label="nominal",width=2.0)
x_nom_sample =  [X_nom_sample[t][1] for t = 1:T]
pltx = plot!(t_span,x_nom_sample,color=:orange,label="sample nominal",width=2.0)
display(pltx)
