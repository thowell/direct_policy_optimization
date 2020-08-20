include("../src/sample_trajectory_optimization.jl")
include("../dynamics/double_integrator.jl")
using Plots

# Horizon
T = 11

tf0 = 1.0
h0 = tf0/(T-1) # timestep

# Hybrid model
Tm = -1
model = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,Tm)
model1 = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,Tm)
model2 = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,Tm)
model3 = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,Tm)
model4 = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,Tm)

# Bounds
# Initial and final states
x1 = [0.0; 0.0]
xT = [1.0; 0.0]

# xl <= x <= xu
xl_traj = [[0.0; -Inf] for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT

# ul <= u <= uu
uu = 5.0
ul = -5.0

# hl <= h <= hu
hu = h0
hl = h0

# Objective
Q = [Diagonal([1.0;1.0]) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [zeros(model.nx) for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([1.0;1.0]) : Diagonal([1.0; 1.0]) for t = 1:T]
# Q_lqr[Tm] = Diagonal([100.0;100.0])
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    general_constraints=true,
                    m_general=model.nx,
                    general_ineq=(1:0))

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)
x_nom = [X_nominal[t][1] for t = 1:T]
v_nom = [X_nominal[t][2] for t = 1:T]

plot(x_nom,v_nom,aspect_ratio=:equal)

K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

# Samples
N = 2*model.nx
models = [model1,model2,model3,model4]
K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 0.0*ones(model.nx)
γ = 1.0

# x1_sample = [[0.0;1.0],[0.0;2.0],[0.0;3.0],[0.0;4.0]]#resample([x1 for i = 1:N],β=β,w=w)
# xT_sample = [[1.0;1.0],[1.0;2.0],[1.0;3.0],[1.0;4.0]]
x1_sample = resample([x1 for i = 1:N],β=β,w=0.1*ones(model.nx))
# x1_sample = [x1 for i = 1:N]
xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

# add "contact" constraint
for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]

    xl_traj_sample[i][T] = xT
    xu_traj_sample[i][T] = xT
end

x1_sample
prob_sample = init_sample_problem(prob,models,
    Q_lqr,R_lqr,H_lqr,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    β=β,w=w,γ=γ,
    policy_constraint=false,
    resample_idx=[])


prob_sample_moi = init_MOI_Problem(prob_sample)

K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=1000,nlp=:SNOPT)
# Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,nlp=:SNOPT7)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
K_sample = [reshape(Z_sample_sol[prob_sample.idx_K[t]],model.nu,model.nx) for t = 1:T-1]
Uw_sample = unpack_disturbance(Z_sample_sol,prob_sample)

# Plot results

# Time
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nominal))s")
display("time (sample nominal): $(sum(H_nom_sample))s")

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nominal...))',
    color=:purple,width=2.0,title="Double Integrator",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
# savefig(plt,joinpath(@__DIR__,"results/double_integrator_control.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="x (nominal)",title="Double Integrator",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="ẋ (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="x (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="ẋ (sample)")
# savefig(plt,joinpath(@__DIR__,"results/double_integrator_state.png"))

# State samples
plt1 = plot(t_sample,hcat(X_nom_sample...)[1,:],color=:red,width=2.0,title="",
    label="",linetype=:steppost);
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1,:],label="",linetype=:steppost);
end

plt2 = plot(t_sample,hcat(X_nom_sample...)[2,:],color=:red,width=2.0,label="",linetype=:steppost);
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt2 = plot!(t_sample,hcat(X_sample[i]...)[2,:],label="",linetype=:steppost);
end
plt12 = plot(plt1,plt2,layout=(2,1),title=["x" "ẋ"],xlabel="time (s)")
# savefig(plt,joinpath(@__DIR__,"results/double_integrator_sample_state.png"))

# Control samples
plt3 = plot(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=2.0,
    title="Control",label="",xlabel="time (s)",linetype=:steppost);
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1,:],label="",
        linetype=:steppost);
end
display(plt3)
# savefig(plt,joinpath(@__DIR__,"results/double_integrator_sample_control.png"))

plot(hcat(Uw_sample[1]...)',linetype=:steppost,labels="",title="Disturbance controls")
plot!(hcat(Uw_sample[2]...)',linetype=:steppost,labels="")
plot!(hcat(Uw_sample[3]...)',linetype=:steppost,labels="")
plot!(hcat(Uw_sample[4]...)',linetype=:steppost,labels="")

# Simulate controllers
using Distributions
model_sim = model
switch=30
T_sim = 10*T+1

model_sim.Tm = convert(Int,(T_sim-1)/2 + 1) + switch

μ = zeros(nx)
Σ = Diagonal(1.0e-32*ones(nx))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(nx)
Σ0 = Diagonal(1.0e-32*ones(nx))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

z0_sim = vec(copy(X_nominal[1]) + w0)
X_nominal
X_nom_sample
sum(H_nominal)
t_sim_nominal = range(0,stop=H_nominal[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)

plt = plot(t_nominal,hcat(X_nominal...)[1:2,:]',legend=:bottom,color=:red,label="",
    width=2.0,xlabel="time (s)",title="Double Integrator (switch=$switch)",ylabel="state")

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nominal,U_nominal,model_sim,Q_lqr,R_lqr,T_sim,H_nominal[1],X_nominal[1],w,_norm=2,
    ul=ul,uu=uu,xl=[0.0;-Inf])
plt = plot!(t_sim_nominal,hcat(z_tvlqr...)[1,:],color=:purple,label="tvlqr",width=2.0)
plt = plot!(t_sim_nominal,hcat(z_tvlqr...)[2,:],color=:purple,label="",width=2.0)

plt = plot(t_sample,hcat(X_nom_sample...)[1:2,:]',legend=:bottom,color=:red,label="",
    width=2.0,xlabel="time (s)",title="Double Integrator (switch=$switch)",ylabel="state")
z_sample, u_sample, J_sample,Jx_sample, Ju_sample = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],X_nom_sample[1],w,_norm=2,
    ul=ul,uu=uu,xl=[0.0;-Inf])
plt = plot!(t_sim_sample,hcat(z_sample...)[1,:],linetype=:steppost,color=:orange,label="sample",width=2.0)
plt = plot!(t_sim_sample,hcat(z_sample...)[2,:],linetype=:steppost,color=:orange,label="",width=2.0)

# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
