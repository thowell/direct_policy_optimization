include("../src/sample_motion_planning.jl")
include("../dynamics/cartpole.jl")
include("../src/general_constraints.jl")
using Plots

model

# Horizon
T = 51

# Bounds

# h = h0 (fixed timestep)
tf0 = 5.0
h0 = tf0/(T-1)
hu = h0
hl = h0

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [0.0; π; 0.0; 0.0]

xl_traj = [-Inf*ones(model.nx) for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT

ul = -10.0
uu = 10.0

# Objective
Q = [t < T ? Diagonal(ones(model.nx)) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal([0.1]) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0;1.0]) : Diagonal(100.0*ones(model_nominal.nx)) for t = 1:T]
R_lqr = [Diagonal([1.0]) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Problem
prob_nominal = init_problem(model.nx,model.nu,T,
                    model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1]
                    )

# MathOptInterface problem
prob_nominal_moi = init_MOI_Problem(prob_nominal)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.1*rand(model_nominal.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob_nominal)

# Solve nominal problem
@time Z_nominal = solve(prob_nominal_moi,copy(Z0),tol=1.0e-5,c_tol=1.0e-5)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob_nominal)

# Time trajectories
t_nominal = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
end

# Plots results

# Control
plt = plot(t_nominal[1:T-1],hcat(U_nominal...)[1:1,:]',color=:purple,width=2.0,
    title="Cartpole",xlabel="time (s)",ylabel="control",label="nominal",
    legend=:topright,linetype=:steppost)

# States
plt = plot(t_nominal,hcat(X_nominal...)[1:4,:]',
    color=:purple,width=2.0,xlabel="time (s)",
    ylabel="state",label=["x (nominal)" "θ (nominal)" "dx (nominal)" "dθ (nominal)"],
    title="Cartpole",legend=:topright)

# Sample
n_policy = model.nu
n_features = 12

function policy(model,K,x,u,x_nom,u_nom)
	px = x[1] + model.l*sin(x[2])
	pz = -model.l*cos(x[2])

	px_nom = x_nom[1] + model.l*sin(x_nom[2])
	pz_nom = -model.l*cos(x_nom[2])

	u_nom - reshape(K,1,12)*[x - x_nom;
							px - px_nom;
							pz - pz_nom
							x.*x - x_nom.*x_nom
							x[1:2].*x[3:4] - x_nom[1:2].*x_nom[3:4]]
end

N = 2*model.nx
models = [model for i = 1:N]

β = 1.0
w = 1.0e-32*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[1]
    xu_traj_sample[i][1] = x1_sample[1]
end

K_lqr = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

K = [rand(n_policy,n_features) for t = 1:T-1]

prob_sample = init_sample_problem(prob_nominal,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
	n_policy=n_policy,
	n_features=n_features,
	xl=xl_traj_sample,
    xu=xu_traj_sample,
    )

prob_sample_moi = init_MOI_Problem(prob_sample)


Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7)
Z_sample_sol = solve(prob_sample_moi,copy(Z_sample_sol),nlp=:SNOPT7)

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
K_sample = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]

t_sample = zeros(T)
for t = 2:T
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

plt_ctrl = plot(title="Cartpole control",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_ctrl = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1:1,:]',label="")
end
plt_ctrl = plot!(t_nominal[1:end-1],hcat(U_nominal...)[1:1,:]',color=:purple,
    width=2.0,label="nominal")
plt_ctrl = plot!(t_sample[1:end-1],hcat(U_nom_sample...)[1:1,:]',color=:orange,
    width=2.0,label="nominal (sample)")
display(plt_ctrl)

plt_state = plot(title="Cartpole w/ state",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_state = plot!(t_sample,hcat(X_sample[i]...)[1:4,:]',label="")
end
plt_state = plot!(t_sample,hcat(X_nominal...)[1:4,:]',color=:purple,
    width=2.0,label=["nominal" "" "" ""])
plt_state = plot!(t_sample,hcat(X_nom_sample...)[1:4,:]',color=:orange,
    width=2.0,label=["nominal (sample)" "" "" ""])
display(plt_state)

# Simulate controllers
using Distributions
model_sim = model
T_sim = 10*T

μ = zeros(nx)
Σ = Diagonal(1.0e-1*ones(nx))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(nx)
Σ0 = Diagonal(1.0e-1*ones(nx))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

z0_sim = vec(copy(X_nominal[1]) + w0)
X_nominal
X_nom_sample
t_sim_nominal = range(0,stop=H_nominal[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)

plt = plot(t_nominal,hcat(X_nominal...)[1:2,:]',legend=:bottom,color=:red,label="",
    width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state")

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K_lqr,
    X_nominal,U_nominal,model_sim,Q_lqr,R_lqr,T_sim,H_nominal[1],X_nominal[1],w,_norm=2,
    ul=ul,uu=uu)
plt = plot!(t_sim_nominal,hcat(z_tvlqr...)[1:2,:]',color=:purple,label=["tvlqr" ""],width=2.0)

plt = plot(t_sample,hcat(X_nom_sample...)[1:2,:]',legend=:bottom,color=:red,label="",
    width=2.0,xlabel="time (s)",title="Cartpole",ylabel="state")
z_sample, u_sample, J_sample,Jx_sample, Ju_sample = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],X_nom_sample[1],w,_norm=2,
    ul=ul,uu=uu,controller=:policy)
plt = plot!(t_sim_sample,hcat(z_sample...)[1:2,:]',linetype=:steppost,color=:orange,label=["sample" ""],width=2.0)

# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
