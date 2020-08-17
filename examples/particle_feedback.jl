include("../src/sample_trajectory_optimization.jl")
include("../dynamics/particle.jl")
include("../src/velocity.jl")
using Plots

# Horizon
tf = 1.0
T = 101
model.Δt = tf/(T-1)

# Initial and final states
x1 = [0.0; 0.0; 10.0]
xT = [1.0; 0.0; 10.0]
v1 = [0.0; 0.0; 0.0]

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
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

# h = h0 (fixed timestep)
hu = model.Δt
hl = model.Δt

# Objective
Q = [t < T ? Diagonal(10.0*ones(model.nx)) : Diagonal(10.0*ones(model.nx)) for t = 1:T]
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
					general_ineq=(1:0),
					v1=v1
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state #TODO clip z
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

x_nom = [X_nom[t][1] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)
plot(x_nom)
plot(z_nom)

model.Δt
X_nom[3]
X_nom[2]
X_nom[1]

t = 5
left_legendre(model,X_nom[t],X_nom[t+1],U_nom[t-2],H_nom[t-2]) - right_legendre(model,X_nom[t+1],X_nom[t+2],U_nom[t-2],H_nom[t-2])

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
Q_lqr = [t < T ? Diagonal([1.0;1.0;1.0]) : Diagonal([1.0; 1.0; 1.0]) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu_ctrl)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Samples
N = 2*model.nx
models = [model for i =1:N]
K0 = [0.01*rand(model.nu_ctrl*(2*model.nx)) for t = 1:T-2]
β = 1.0
w = 1.0e-8*ones(model.nx)
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

# policy
function policy(model::Particle,K,x1,x2,x3,u,h,x1_nom,x2_nom,x3_nom,u_nom,h_nom)
	v = left_legendre(model,x2,x3,u,h)
	v_nom = left_legendre(model,x2_nom,x3_nom,u_nom,h_nom)
	u_nom[model.idx_u] - reshape(K,model.nu_ctrl,(2*model.nx))*[x3 - x3_nom; v - v_nom]
end

prob_sample = init_sample_problem(prob,models,x1_sample,
    Q_lqr,R_lqr,H_lqr,
	u_policy=model.idx_u,
	nK=model.nu_ctrl*(2*model.nx),
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
    m_sample_general=N*model.nx,
    sample_general_ineq=(1:0),
	general_objective=true)
prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K0,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100)
Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,max_iter=100)

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

pltu = plot(label="",xlabel="time (s)",ylabel="u",title="Particle",
	legend=:bottomright)
t_span = range(0,stop=model.Δt*(T-1),length=T)
for i = 1:N
	pltu = plot!(t_span[1:end-1],hcat(U_sample[i]...)[1:3,:]',label="")
end

pltu = plot(t_span[1:end-2],hcat(U_nom...)[1:3,:]',color=:purple,label="nominal",width=2.0)
pltu = plot!(t_span[1:end-2],hcat(U_nom_sample...)[1:3,:]',color=:orange,label="sample nominal",width=2.0)
display(pltu)

K_nom_sample = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-2]

# get x1,x2 that match velocity and x3
X_nom = X_nom_sample
U_nom = U_nom_sample
H_nom = H_nom_sample

include("../src/velocity.jl")
T_scale = 5
T_sim = T_scale*T

tf = sum(H_nom)
times = [(t-1)*H_nom[t] for t = 1:T-2]
t_sim = range(0,stop=tf,length=T_sim)
dt_sim = tf/(T_sim-1)


# xl <= x <= xu
xu_vel = [Inf*ones(model.nx) for t = 1:3]
xl_vel = [-Inf*ones(model.nx) for t = 1:3]
xu_vel[2] = X_nom[2]
xl_vel[2] = X_nom[2]

# ul <= u <= uu
uu_vel = Inf*ones(model.nu)
ul_vel = zeros(model.nu)

uu_vel[model.idx_u] .= Inf
ul_vel[model.idx_u] .= -Inf

general_constraint=true
m_general=model.nx

# Problem
prob_vel = init_problem(model.nx,model.nu,3,model,penalty_obj,
	                    xl=xl_vel,
	                    xu=xu_vel,
	                    ul=[ul_vel],
	                    uu=[uu_vel],
	                    hl=[dt_sim],
	                    hu=[dt_sim],
						general_constraints=general_constraint,
						m_general=m_general,
						general_ineq=(1:0),
	                    v1=v1)

# MathOptInterface problem
prob_vel_moi = init_MOI_Problem(prob_vel)

# Pack trajectories into vector
Z0_vel = pack([X_nom[1],X_nom[2],X_nom[3]],[U_nom[1]],dt_sim,prob_vel)

@time Z_vel_sol = solve(prob_vel_moi,copy(Z0_vel),tol=1.0e-6,c_tol=1.0e-6)
X_vel_sol, U_vel_sol, H_vel_sol = unpack(Z_vel_sol,prob_vel)

norm(X_vel_sol[2] - X_nom[2])
v_vel = legendre(model,X_vel_sol[2],X_vel_sol[3],U_vel_sol[1],H_vel_sol[1])
norm(v1 - v_vel)

include("../src/simulate.jl")

X_sim_policy, U_sim_policy, dt_sim_policy = simulate_policy(model,
	X_nom_sample,U_nom_sample,H_nom_sample,K_nom_sample,T_sim,X_vel_sol[1],X_vel_sol[2],
	α=1000.0,slack_tol=1.0e-6,tol=1.0e-6,c_tol=1.0e-6)
model.Δt = dt_sim_policy

X_sim_nom, U_sim_nom, dt_sim_nom = simulate_nominal(model,
	X_nom_sample,U_nom_sample,H_nom_sample,K_nom_sample,T_sim,dt_sim,
	X_vel_sol[1],X_vel_sol[2],
	α=1000.0,slack_tol=1.0e-6,tol=1.0e-6,c_tol=1.0e-6)
X_sim_policy[2]
X_nom[2]

norm(X_sim_policy[2] - X_nom_sample[2])
t_sim = zeros(T_sim-2)
for t = 2:T_sim-2
    t_sim[t] = t_sim[t-1] + dt_sim_nom
end
sum(H_nom_sample)
t_sim[end]
# visualize!(vis,model,[X_sim_policy, X_sim_nom],color=[RGBA(0, 1, 0, 1.0),RGBA(1, 0, 0, 1.0)])

t_nominal = zeros(T-2)
t_sample = zeros(T-2)
for t = 2:T-2
	t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

plt_track = plot(t_sample,hcat(X_nom_sample[2:T-1]...)',color=:red,label=["nominal" "" ""],
	title="Particle tracking performance ($(T_scale)T)",
	xlabel="time (s)",
	ylabel="state",
	legend=:right)
plt_track = plot!(t_sim,hcat(X_sim_nom[2:T_sim-1]...)',color=:purple,label=["open loop" "" ""])
plt_track = plot!(t_sim,hcat(X_sim_policy[2:T_sim-1]...)',color=:orange,label=["policy" "" ""])

# savefig(plt_track,joinpath(@__DIR__,"results/particle_tracking_$(T_scale)T.png"))
