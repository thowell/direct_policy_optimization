include("../src/sample_trajectory_optimization.jl")
include("../dynamics/cartpole.jl")
using Plots

model = model_friction
model.μ = 0.01

# Horizon
T = 25 # 51

# Bounds

# h = h0 (fixed timestep)
tf0 = 2.5
h0 = tf0/(T-1)
hu = h0
hl = h0

# Initial and final states
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [0.0; π; 0.0; 0.0]

# Objective
Q = [t < T ? Diagonal(ones(model.nx)) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal([0.1,0.0,0.0,0.0,0.0,0.0,0.0]) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])
penalty_obj = PenaltyObjective(α_cartpole_friction)

multi_obj = MultiObjective([obj,penalty_obj])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0;1.0]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal([0.1,0.0,0.0,0.0,0.0,0.0,0.0]) for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    ul=[ul_friction for t=1:T-1],
                    uu=[uu_friction for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true,
                    stage_constraints=true,
                    m_stage=[m_stage_friction for t=1:T-1],
                    stage_ineq=[stage_friction_ineq for t=1:T-1])

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

# Time trajectories
t_nominal = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
end

# Plots results
S_nominal = [U_nominal[t][7] for t=1:T-1]
b_nominal = [U_nominal[t][2] - U_nominal[t][3] for t=1:T-1]

# Control
plt = plot(t_nominal[1:T-1],hcat(U_nominal...)[1:1,:]',color=:purple,width=2.0,
    title="Cartpole",xlabel="time (s)",ylabel="control",label="nominal",
    legend=:topright,linetype=:steppost)

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",
    ylabel="state",label="x (nominal)",title="Cartpole",legend=:topright)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],
    color=:purple,width=2.0,label="θ (nominal)")
plt = plot!(t_nominal,hcat(X_nominal...)[3,:],
    color=:purple,width=2.0,label="dx (nominal)")
plt = plot!(t_nominal,hcat(X_nominal...)[4,:],
    color=:purple,width=2.0,label="dθ (nominal)")

# Sample
N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-8*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)
K = TVLQR_policy(model_nominal,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr,u_ctrl=(1:1))

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    u_ctrl=(1:1),
    general_objective=true)
prob_sample_moi = init_MOI_Problem(prob_sample)

Ū_nominal = deepcopy(U_nominal)
for t=1:T-1
    Ū_nominal[t][2:7] = 0.001*rand(model.nu-1)
end
Z0_sample = pack(X_nominal,Ū_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample))

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

# Plot
t_sample = zeros(T)
for t = 2:T
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

plt_ctrl = plot(title="Cartpole w/ friction control",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_ctrl = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1:1,:]',label="")
end
plt_ctrl = plot!(t_sample[1:end-1],hcat(U_nom_sample...)[1:1,:]',color=:red,
    width=2.0,label="nominal")
plt_ctrl = plot!(t_nominal[1:end-1],hcat(U_nominal...)[1:1,:]',color=:purple,
    width=2.0,label="nominal (original)")
display(plt_ctrl)

plt_state = plot(title="Cartpole w/ friction state",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_state = plot!(t_sample,hcat(X_sample[i]...)[1:2,:]',label="")
end
plt_state = plot!(t_sample,hcat(X_nom_sample...)[1:1,:]',color=:red,
    width=2.0,label="nominal")
display(plt_state)
