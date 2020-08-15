include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
include("../src/loop.jl")
using Plots

# Horizon
T = 51
Tm = convert(Int,(T-3)/2 + 3)

tf = 1.0
model.Δt = tf/(T-1)

zh = 0.5
# Initial and final states
x1 = [0.0, 0.0, zh]
xM = [0.5, 0.0, 0.0]
xT = [1.0, 0.0, zh]

# Bounds
# xl <= x <= xu
xu = Inf*ones(model.nx)
xl = -Inf*ones(model.nx)
xu_traj = [xu for t=1:T]
xl_traj = [xl for t=1:T]

xu_traj[3] = x1
# xu_traj[Tm] = xM
# xu_traj[T] = xT

xl_traj[3] = x1
# xl_traj[Tm] = xM
# xl_traj[T] = xT

# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

# h = h0 (fixed timestep)
hu = model.Δt
hl = model.Δt

# Objective
Q = [t < T ? Diagonal([1.0,1.0,1.0]) : Diagonal([1.0,1.0,1.0]) for t = 1:T]
R = [Diagonal([1.0e-1,1.0e-1,1.0e-1]) for t = 1:T-2]
c = 0.0

X_ref = linear_interp(x1,xT,T)
X_ref[Tm] = xM

obj = QuadraticTrackingObjective(Q,R,c,
    [X_ref[t] for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])
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
					m_general=model.nx-1+model.nx,
					general_ineq=(1:0),
                    contact_sequence=true,
					T_contact_sequence=[Tm])

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
@assert norm(ϕ_func(model,X_nom[Tm])) < 1.0e-5
@assert norm(X_nom[3][2:end] - X_nom[T][2:end]) < 1.0e-5
plot(λ_nom,linetype=:steppost)

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
# visualize!(vis,model,[X_nom])

# samples
Q_lqr = [t < T ? Diagonal([10.0;10.0;10.0]) : Diagonal([10.0; 10.0; 10.0]) for t = 1:T]
R_lqr = [Diagonal(1.0e-2*ones(model.nu_ctrl)) for t = 1:T-2]
H_lqr = [0.0 for t = 1:T-1]

# Samples
N = 2*model.nx
models = [model for i =1:N]
K0 = [rand(model.nu_ctrl*(model.nx + 2*model.nc)) for t = 1:T-2]
β = 1.0
w = 1.0e-3*ones(model.nx)
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

function policy(model::Particle,K,x1,x2,x3,u,h,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
	v = (x3 - x2)/h[1]
	v_nom = (x3_nom - x2_nom)/h_nom[1]
	u_nom[model.idx_u] - reshape(K,model.nu_ctrl,model.nx + 2*model.nc)*[x3 - x3_nom;
											   ϕ_func(model,x3) - ϕ_func(model,x3_nom);
											   u[model.idx_λ] - u_nom[model.idx_λ]]
end

prob_sample = init_sample_problem(prob,models,x1_sample,
    Q_lqr,R_lqr,H_lqr,
	u_policy=model.idx_u,
	nK=length(model.idx_u)*(model.nx + 2*model.nc),
    β=β,w=w,γ=γ,
    disturbance_ctrl=true,
    α=1.0,
	ul=ul_traj_sample,
	uu=uu_traj_sample,
	xl=xl_traj_sample,
	xu=xu_traj_sample,
	hl=hl_traj_sample,
	hu=hu_traj_sample,
    sample_general_constraints=true,
    m_sample_general=N*(model.nx-1 + model.nx),
    sample_general_ineq=(1:0),
	general_objective=true,
	sample_contact_sequence=true,
	T_sample_contact_sequence=[[Tm-3],[Tm-2],[Tm-1],[Tm+1],[Tm+2],[Tm+3]])

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K0,prob_sample)

# Solve
#NOTE: run multiple times to get good solution
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=1000)
Z_sample_sol = solve(prob_sample_moi,Z_sample_sol)

X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

t_nominal = zeros(T-2)
t_sample = zeros(T-2)
for t = 2:T-2
	t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

s = [U_nom_sample[t][model.idx_s] for t = 1:T-2]
@assert sum(s) < 1.0e-5

pltz = plot(label="",xlabel="time (s)",ylabel="z",title="Particle",
	legend=:topright)
for i = 1:N
	z_sample = [X_sample[i][t][3] for t = 3:T]
	pltz = plot!(t_sample,z_sample,label="")
end
pltz = plot!(t_nominal,z_nom[3:T],color=:purple,label="nominal",width=2.0)
z_nom_sample =  [X_nom_sample[t][3] for t = 1:T]
pltz = plot!(t_sample,z_nom_sample[3:T],color=:orange,label="sample nominal",width=2.0,legend=:bottomright)
display(pltz)
# savefig(pltz,joinpath(@__DIR__,"results/particle_soft_contact_v2_z_T$T.png"))

pltx = plot(label="",xlabel="time (s)",ylabel="x",title="Particle",
	legend=:bottomright)
for i = 1:N
	x_sample = [X_sample[i][t][1] for t = 1:T]
	pltx = plot!(t_sample,x_sample[3:T],label="")
end
pltx = plot!(t_nominal,x_nom[3:T],color=:purple,label="nominal",width=2.0)
x_nom_sample =  [X_nom_sample[t][1] for t = 1:T]
pltx = plot!(t_sample,x_nom_sample[3:T],color=:orange,label="sample nominal",width=2.0)
display(pltx)
# savefig(pltz,joinpath(@__DIR__,"results/particle_soft_contact_v2_z_T$T.png"))

# visualize!(vis,model,[X_nom, X_nom_sample],
# 	color=[RGBA(1, 0, 0, 1.0), RGBA(0, 1, 0, 1.0)])

K_nom_sample = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-2]

# simulate policy
include("../src/simulate.jl")
T_scale =
T_sim = T_scale*T

X_sim_policy, U_sim_policy, dt_sim_policy = simulate_policy(model,
	X_nom_sample,U_nom_sample,H_nom_sample,K_nom_sample,T_sim,
	α=100.0,slack_tol=1.0e-5,tol=1.0e-5,c_tol=1.0e-5)
model.Δt = dt_sim_policy

X_sim_nom, U_sim_nom, dt_sim_nom = simulate_nominal(model,
	X_nom_sample,U_nom_sample,H_nom_sample,K_nom_sample,T_sim,
	α=100.0,slack_tol=1.0e-6,tol=1.0e-6,c_tol=1.0e-6)

t_sim = zeros(T_sim-2)
for t = 2:T_sim-2
    t_sim[t] = t_sim[t-1] + dt_sim_policy
end
sum(H_nom_sample)
# visualize!(vis,model,[X_sim_policy, X_sim_nom],color=[RGBA(0, 1, 0, 1.0),RGBA(1, 0, 0, 1.0)])

plt_track = plot(t_sample,hcat(X_nom_sample[3:T]...)',color=:red,label=["nominal" "" ""],
	title="Particle tracking performance ($(T_scale)T)",
	xlabel="time (s)",
	ylabel="state",
	legend=:topleft)
plt_track = plot!(t_sim,hcat(X_sim_nom[3:T_sim]...)',color=:purple,label=["open loop" "" ""])
plt_track = plot!(t_sim,hcat(X_sim_policy[3:T_sim]...)',color=:orange,label=["policy" "" ""])

# savefig(plt_track,joinpath(@__DIR__,"results/particle_tracking_$(T_scale)T.png"))
