include("../src/sample_trajectory_optimization.jl")
include("../dynamics/hopper.jl")
# include("../src/loop_up.jl")
using Plots

# Horizon
T = 21
Tm = convert(Int,(T-3)/2 + 3)

tf = 1.0
model.Δt = tf/(T-1)

zh = 0.1
# Initial and final states
x1 = [0., model.r+zh, model.r, 0., 0.]
# xM = [0., 0.5*model.r, 0.5*model.r, 0., 0.]
xT = [1.0, model.r+zh, model.r, 0., 0.]

# Bounds
# xl <= x <= xu
xu_traj = [model.qU for t=1:T]
xl_traj = [model.qL for t=1:T]

xu_traj[2] = x1
xu_traj[T] = xT


xl_traj[2] = x1
xl_traj[T] = xT

# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

ul_traj = [ul for t = 1:T-2]
uu_traj = [uu for t = 1:T-2]

# h = h0 (fixed timestep)
hu = model.Δt
hl = model.Δt

# Objective
Q = [t != Tm ? Diagonal([1.0,1.0,1.0,0.1,0.1]) : 10.0*Diagonal([1.0,1.0,1.0,0.1,0.1]) for t = 1:T]
R = [Diagonal([1.0,1.0]) for t = 1:T-2]
c = 0.0

obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t = 1:T],[zeros(model.nu_ctrl) for t=1:T-2])
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
					general_constraints=false,
					m_general=0*2*model.nx,
					general_ineq=(1:0),
					contact_sequence=true,
					T_contact_sequence=[Tm]
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = [linear_interp(x1,xM,Tm-1)[1:end-1]...,linear_interp(xM,x1,Tm-1)...]
X0[Tm] = xM
U0 = [0.001*rand(model.nu) for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
@time Z_nominal = solve(prob_moi,copy(Z_nominal))

X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
plot(hcat(U_nom...)[model.idx_u,:]',linetype=:steppost)
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]

x_nom = [X_nom[t][1] for t = 1:T]
y_nom = [X_nom[t][2] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]
norm(s_nom,Inf)
t_nom = zeros(T-2)
for t = 2:T-2
    t_nom[t] = t_nom[t-1] + H_nom[t-1]
end
ϕ_func(model,X_nom[Tm])
@assert norm(s_nom,Inf) < 1.0e-5
@assert norm(ϕ_func(model,X_nom[Tm])) < 1.0e-5
@assert norm(X_nom[2] - X_nom[T]) < 1.0e-5
λ = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
b = [U_nom[t][model.idx_b] for t = 1:T-2]
ψ = [U_nom[t][model.idx_ψ[1]] for t = 1:T-2]
η = [U_nom[t][model.idx_η] for t = 1:T-2]

plot(λ,linetype=:steppost)
plot!(hcat(b...)',linetype=:steppost)
plot!(ψ,linetype=:steppost)
plot!(hcat(η...)',linetype=:steppost)

plot([ϕ_func(model,X_nom[t])[1] for t = 3:T],linetype=:steppost)

sum(H_nom)

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
# model.Δt = H_nom[1]
# visualize!(vis,model,X_nom)


plot(hcat(X_nom...)')
plot(hcat(U_nom...)[1:2,:]',linetype=:steppost)


# samples
Q_lqr = [t != -1 ? Diagonal([1.0;1.0;1.0;0.1;0.1]) : Diagonal([1.0;1.0;1.0;0.1;0.1]) for t = 1:T]
R_lqr = [Diagonal([1.0e-2; 1.0e-2]) for t = 1:T-2]
H_lqr = [0.0 for t = 1:T-1]

# Samples
N = 2*model.nx
models = [model for i =1:N]
K0 = [0.0*rand(model.nu_ctrl*(model.nx + 2*model.nc)) for t = 1:T-2]
β = 1.0
w = 1.0e-32*ones(model.nx)
γ = 1.0
x1_sample = [x1 for i = 1:N]#resample([x1 for i = 1:N],β=β,w=1.0e-3*ones(model.nx))

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

function policy(model::Hopper,K,x1,x2,x3,u,h,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
	# v = legendre(model,x1,x2,x3,u,h)
	# v_nom = legendre(model,x1_nom,x2_nom,x3_nom,u_nom,h_nom)

	# u_nom[model.idx_u] - reshape(K,model.nu_ctrl,model.nx + 2*model.nc)*[(x3 - x3_nom);
	# 										    ϕ_func(model,x3) - ϕ_func(model,x3_nom);
	# 										    u[model.idx_λ] - u_nom[model.idx_λ]]
	u[model.idx_u]
end

prob_sample = init_sample_problem(prob,models,x1_sample,
    Q_lqr,R_lqr,H_lqr,
	resample=true,
	u_policy=model.idx_u,
	nK = model.nu_ctrl*(model.nx + 2*model.nc),
    β=β,w=w,γ=γ,
    disturbance_ctrl=false,
    α=1.0e-1,
	ul=ul_traj_sample,
	uu=uu_traj_sample,
	xl=xl_traj_sample,
	xu=xu_traj_sample,
	hl=hl_traj_sample,
	hu=hu_traj_sample,
    sample_general_constraints=false,
    m_sample_general=0*N*(2*model.nx),
    sample_general_ineq=(1:0),
	general_objective=true,
	sample_contact_sequence=true,
	T_sample_contact_sequence=[[Tm],[Tm],[Tm],[Tm],[Tm],[Tm],[Tm],[Tm],[Tm],[Tm]]
	)

prob_sample_moi = init_MOI_Problem(prob_sample)

for t = 1:T-2
	prob_sample_moi.primal_bounds[1][prob_sample.idx_K[t]] .= 0.0
	prob_sample_moi.primal_bounds[2][prob_sample.idx_K[t]] .= 0.0
end

Z0_sample = pack(X_nom,U_nom,H_nom[1],K0,prob_sample)

# Solve
#NOTE: run multiple times to get good solution
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100)
Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,max_iter=500)

X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

s = [U_nom_sample[t][model.idx_s] for t = 1:T-2]
@assert norm(s,Inf) < 1.0e-5

t_sample = zeros(T-2)
for t = 2:T-2
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

pltz = plot(label="",xlabel="time (s)",ylabel="z",title="Hopper",
	legend=:topright)

for i = 1:N
	z_sample = [X_sample[i][t][3] for t = 1:T]
	pltz = plot!(t_sample,z_sample[3:T],label="")
end
pltz = plot!(t_nom,z_nom[3:T],color=:purple,label="nominal",width=2.0)
z_nom_sample =  [X_nom_sample[t][3] for t = 1:T]
pltz = plot!(t_sample,z_nom_sample[3:T],color=:orange,label="sample nominal",width=2.0,legend=:bottomright)
display(pltz)
savefig(pltz,joinpath(@__DIR__,"results/hopper_z_T$T.png"))

pltx = plot(label="",xlabel="time (s)",ylabel="x",title="Hopper",
	legend=:bottomright)
t_span = range(0,stop=model.Δt*(T-1),length=T)
for i = 1:N
	x_sample = [X_sample[i][t][1] for t = 1:T]
	pltx = plot!(t_sample,x_sample[3:T],label="")
end
pltx = plot!(t_nom,x_nom[3:T],color=:purple,label="nominal",width=2.0)
x_nom_sample =  [X_nom_sample[t][1] for t = 1:T]
pltx = plot!(t_sample,x_nom_sample[3:T],color=:orange,label="sample nominal",width=2.0)
display(pltx)
savefig(pltx,joinpath(@__DIR__,"results/hopper_x_T$T.png"))

plt_sdf = plot(label="",xlabel="time (s)",ylabel="sdf",title="Hopper",
	legend=:bottomright)
for i = 1:N
	sdf_sample = [ϕ_func(model,X_sample[i][t])[1] for t = 1:T]
	plt_sdf = plot!(t_sample,sdf_sample[3:T],label="")
end
plt_sdf = plot!(t_nom,[ϕ_func(model,X_nom[t])[1] for t = 3:T],color=:purple,label="nominal",width=2.0)
plt_sdf = plot!(t_sample,[ϕ_func(model,X_nom_sample[t])[1] for t = 3:T],color=:orange,label="sample nominal",width=2.0)
display(plt_sdf)
savefig(plt_sdf,joinpath(@__DIR__,"results/hopper_sdf_T$T.png"))

plt_ctrl = plot(label="",xlabel="time (s)",ylabel="ctrl",title="Hopper",
	legend=:bottomright)
for i = 1:N
	plt_ctrl = plot!(t_sample,hcat(U_sample[i]...)[1:model.nu_ctrl,:]',linetype=:steppost,label="")
end
plt_ctrl = plot!(t_nom,hcat(U_nom...)[1:model.nu_ctrl,:]',linetype=:steppost,color=:purple,label="nominal",width=2.0)
plt_ctrl = plot!(t_sample,hcat(U_nom_sample...)[1:model.nu_ctrl,:]',linetype=:steppost,color=:orange,label="sample nominal",width=2.0)
display(plt_ctrl)
savefig(plt_ctrl,joinpath(@__DIR__,"results/hopper_ctrl_T$T.png"))

visualize!(vis,model,X_nom_sample)

K_nom_sample = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-2]

# get x1,x2 that match velocity and x3
X_nom = X_nom_sample
U_nom = U_nom_sample
H_nom = H_nom_sample

include("../src/velocity.jl")
T_scale = 1
T_sim = T_scale*T

tf = sum(H_nom_sample)
times = [(t-1)*H_nom_sample[t] for t = 1:T-2]
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)

# xl <= x <= xu
xu_vel = [Inf*ones(model.nx) for t = 1:3]
xl_vel = [-Inf*ones(model.nx) for t = 1:3]
xu_vel[2] = X_nom[2]
xl_vel[2] = X_nom[2]

# ul <= u <= uu
v1_sample = left_legendre(model,X_nom_sample[2],X_nom_sample[3],U_nom_sample[1],H_nom_sample[1])
general_constraint=true
m_general=model.nx

# Problem
prob_vel = init_problem(model.nx,model.nu,3,model,penalty_obj,
	                    xl=xl_vel,
	                    xu=xu_vel,
	                    ul=[ul],
	                    uu=[uu],
	                    hl=[dt_sim],
	                    hu=[dt_sim],
						general_constraints=general_constraint,
						m_general=m_general,
						general_ineq=(1:0),
	                    v1=v1_sample)

# MathOptInterface problem
prob_vel_moi = init_MOI_Problem(prob_vel)

# Pack trajectories into vector
Z0_vel = pack([X_nom_sample[1],X_nom_sample[2],X_nom_sample[3]],[U_nom_sample[1]],dt_sim,prob_vel)

@time Z_vel_sol = solve(prob_vel_moi,copy(Z0_vel),tol=1.0e-6,c_tol=1.0e-6)
X_vel_sol, U_vel_sol, H_vel_sol = unpack(Z_vel_sol,prob_vel)

norm(X_vel_sol[2] - X_nom[2])
v_vel = left_legendre(model,X_vel_sol[2],X_vel_sol[3],U_vel_sol[1],H_vel_sol[1])
norm(v1_sample - v_vel)

include("../src/simulate.jl")

X_sim_policy, U_sim_policy, dt_sim_policy = simulate_policy(model,
	X_nom_sample,U_nom_sample,H_nom_sample,K_nom_sample,T_sim,X_vel_sol[1],X_vel_sol[2],
	α=100.0,slack_tol=1.0e-5,tol=1.0e-5,c_tol=1.0e-5)
model.Δt = dt_sim_policy


v1_sim = (X_sim_policy[2] - X_sim_policy[1])/dt_sim_policy
v1_sample

X_sim_nom, U_sim_nom, dt_sim_nom = simulate_nominal(model,
	X_nom_sample,U_nom_sample,H_nom_sample,K_nom_sample,T_sim,dt_sim,
	X_vel_sol[1],X_vel_sol[2],
	α=100.0,slack_tol=1.0e-5,tol=1.0e-5,c_tol=1.0e-5)

t_sim = zeros(T_sim-2)
for t = 2:T_sim-2
    t_sim[t] = t_sim[t-1] + dt_sim_nom
end

# visualize!(vis,model,[X_sim_policy, X_sim_nom],color=[RGBA(0, 1, 0, 1.0),RGBA(1, 0, 0, 1.0)])

t_nominal = zeros(T-2)
t_sample = zeros(T-2)
for t = 2:T-2
	t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

plt_track = plot([-H_nom[1],t_sample...],hcat(X_nom_sample[2:T]...)',color=:red,label=["nominal" "" ""],
	title="Hopper tracking performance ($(T_scale)T)",
	xlabel="time (s)",
	ylabel="state",
	legend=:left)

# plt_track = plot!([-dt_sim_nom,t_sim...],hcat(X_sim_nom[2:T_sim]...)',color=:purple,label=["open loop" "" ""])
plt_track = plot!([-dt_sim_policy,t_sim...],hcat(X_sim_policy[2:T_sim]...)',color=:orange,label=["policy" "" ""])

# savefig(plt_track,joinpath(@__DIR__,"results/particle_tracking_$(T_scale)T.png"))
