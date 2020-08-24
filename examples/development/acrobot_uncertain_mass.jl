include("../src/sample_motion_planning.jl")
include("../dynamics/acrobot.jl")
using Plots

# Horizon
T = 51

# Bounds

# ul <= u <= uu
uu = 20.0
ul = -20.0

# hl <= h <= hu
tf0 = 10.0
h0 = tf0/(T-1) # timestep

hu = 10.0*h0
hl = 0.0*h0

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [π; 0.0; 0.0; 0.0]

xl = -Inf*ones(model.nx)
xu = Inf*ones(model.nx)
# xl[2] = -π
# xu[2] = π
# xl[3] = -5.0
# xu[3] = 5.0
# xl[4] = -5.0
# xu[4] = 5.0
xl_traj = [xl for t = 1:T]
xu_traj = [xu for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT

# Objective
Q = [t<T ? 0.0*Diagonal(ones(model.nx)) : 0.0*Diagonal(10.0*ones(model.nx)) for t = 1:T]
R = [0.0*Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 1.0

x_ref = linear_interp(x1,xT,T)
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal(10.0*ones(model.nx)) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [1.0 for t = 1:T-1]

# Problem
prob_nom = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1]
                    )

# MathOptInterface problem
prob_nom_moi = init_MOI_Problem(prob_nom)

# Initialization
X0 = linear_interp(x1,xT,T) # linear interpolation for states
U0 = [rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob_nom)

# Solve nominal problem
@time Z_nominal = solve(prob_nom_moi,copy(Z0),nlp=:SNOPT7)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob_nom)
sum(H_nominal)
t_nominal = zeros(T)
for t = 1:T-1
    t_nominal[t+1] = t_nominal[t] + H_nominal[t]
end

plot(t_nominal,hcat(X_nominal...)',xlabel="time step",width=2.0)
plot(t_nominal[1:T-1],hcat(U_nominal...)',xlabel="time step",linetype=:steppost,width=2.0,color=:red,label="")

# Samples
N = 2*model.nx
# Models
model1 = Acrobot(1.0,1.0,1.0,0.5,1.1,1.0,1.0,0.5,9.81,nx,nu)
model2 = Acrobot(1.0,1.0,1.0,0.5,1.075,1.0,1.0,0.5,9.81,nx,nu)
model3 = Acrobot(1.0,1.0,1.0,0.5,1.05,1.0,1.0,0.5,9.81,nx,nu)
model4 = Acrobot(1.0,1.0,1.0,0.5,1.025,1.0,1.0,0.5,9.81,nx,nu)
model5 = Acrobot(1.0,1.0,1.0,0.5,0.975,1.0,1.0,0.5,9.81,nx,nu)
model6 = Acrobot(1.0,1.0,1.0,0.5,0.95,1.0,1.0,0.5,9.81,nx,nu)
model7 = Acrobot(1.0,1.0,1.0,0.5,0.925,1.0,1.0,0.5,9.81,nx,nu)
model8 = Acrobot(1.0,1.0,1.0,0.5,0.9,1.0,1.0,0.5,9.81,nx,nu)
models = [model1,model2,model3,model4,model5,model6,model7,model8]
# models = [model for i = 1:N]
β = 3.0
w = 0.0*ones(model.nx)
γ = 1.0
x1_sample = [x1 for i = 1:N]#
# x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[xl for t = 1:T] for i = 1:N]
xu_traj_sample = [[xu for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[1]
    xu_traj_sample[i][1] = x1_sample[1]

    # xl_traj_sample[i][T] = xT
    # xu_traj_sample[i][T] = xT
end

K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

prob_sample = init_sample_problem(prob_nom,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
   )

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,nlp=:SNOPT,time_limit=300)
# Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,nlp=:SNOPT,time_limit=600)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
K_sample = [reshape(Z_sample_sol[prob_sample.idx_K[t]],model.nu,model.nx) for t = 1:T-1]

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
    color=:purple,width=2.0,title="Acrobot",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/acrobot_control_mass.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ1 (nominal)",title="Acrobot",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="θ2 (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ1 (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="θ2 (sample)")
savefig(plt,joinpath(@__DIR__,"results/acrobot_state_mass.png"))


# State samples
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt = plot!(t_sample,hcat(X_sample[i]...)[1:2,:]',label="");
end
display(plt)

# Control samples
plt3 = plot(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=2.0,
    title="Control",label="",xlabel="time (s)");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1,:],label="");
end
display(plt3)
savefig(plt,joinpath(@__DIR__,"results/acrobot_sample_control.png"))

# vis = Visualizer()
# open(vis)
# visualize!(vis,model,[X_nominal,X_nom_sample,X_sample...],
#     color=[RGBA(128/255,0,128/255,1.0),RGBA(255/255,165/255,0,1.0),[RGBA(1,0,0,0.5) for i = 1:N]...],Δt=h0)

# simulate controller
using Distributions
model_sim = model
model_sim.m2 = 0.9
T_sim = 10*T
W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w0 = rand(W0,1)
z0_sim = vec(copy(x1) + 1.0*w0[:,1])

t_nominal = range(0,stop=h0*(T-1),length=T)
t_sim = range(0,stop=h0*(T-1),length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nominal,U_nominal,model_sim,Q_lqr,R_lqr,
    T_sim,H_nominal[1],z0_sim,w,ul=ul,uu=uu)

pltx1 = plot(t_nominal,hcat(X_nominal...)[1:2,:]',color=:red,label=["nominal" ""],
    xlabel="time (s)",title="Acrobot")
pltx1 = plot!(t_sim,hcat(z_tvlqr...)[1:2,:]',color=:purple,label=["tvlqr" ""],
    legend=:top)

pltu1 = plot(t_nominal[1:end-1],hcat(U_nominal...)[1:1,:]',color=:red,label=["nominal" ""],
    xlabel="time (s)",title="Acrobot")
pltu1 = plot!(t_sim[1:end-1],hcat(u_tvlqr...)[1:1,:]',color=:purple,label=["tvlqr" ""],
    legend=:top)

z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,
    T_sim,H_nom_sample[1],z0_sim,w,ul=ul,uu=uu)
pltx2 = plot(t_nominal,hcat(X_nom_sample...)[1:2,:]',color=:red,label=["nominal (sample)" ""],
    xlabel="time (s)",title="Acrobot",legend=:top)
pltx2 = plot!(t_sim,hcat(z_sample...)[1:2,:]',color=:orange,label=["sample" ""],width=2.0)

pltu2 = plot(t_nominal[1:end-1],hcat(U_nom_sample...)[1:1,:]',color=:red,label=["nominal (sample)" ""],
    xlabel="time (s)",title="Acrobot")
pltu2 = plot!(t_sim[1:end-1],hcat(u_sample...)[1:1,:]',color=:orange,label=["(sample)" ""],
    legend=:top)


# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample

# vis = Visualizer()
# open(vis)
# dt_sim = sum(H_nom_sample)/(T_sim-1)
# visualize!(vis,model,[z_sample],Δt=dt_sim)
