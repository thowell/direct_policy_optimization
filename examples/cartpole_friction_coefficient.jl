include("../src/sample_trajectory_optimization.jl")
include("../dynamics/cartpole.jl")
using Plots

α_cartpole_friction = 100.0
μ0 = 0.2

model_nominal = CartpoleFriction(1.0,0.2,0.5,9.81,0.0,
    nx_friction,nu_friction,nu_policy_friction)
model_friction = CartpoleFriction(1.0,0.2,0.5,9.81,μ0,
    nx_friction,nu_friction,nu_policy_friction)

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

# Objective
Q = [t < T ? Diagonal(ones(model_nominal.nx)) : Diagonal(zeros(model_nominal.nx)) for t = 1:T]
R = [Diagonal([0.1,0.0,0.0,0.0,0.0,0.0,0.0]) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model_nominal.nu) for t=1:T])
penalty_obj = PenaltyObjective(α_cartpole_friction)

multi_obj = MultiObjective([obj,penalty_obj])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0;1.0]) : Diagonal(100.0*ones(model_nominal.nx)) for t = 1:T]
R_lqr = [Diagonal([0.1,0.0,0.0,0.0,0.0,0.0,0.0]) for t = 1:T-1]
H_lqr = [0.0 for t = 1:T-1]

# Problem
prob_nominal = init_problem(model_nominal.nx,model_nominal.nu,T,
                    model_nominal,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul_friction for t=1:T-1],
                    uu=[uu_friction for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    general_constraints=true,
                    m_general=m_stage_friction*(T-1),
                    general_ineq=vcat([(t-1)*m_stage_friction .+ stage_friction_ineq for t=1:T-1]...))

prob_friction = init_problem(model_friction.nx,model_friction.nu,T,
                    model_friction,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul_friction for t=1:T-1],
                    uu=[uu_friction for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    general_constraints=true,
                    m_general=m_stage_friction*(T-1),
                    general_ineq=vcat([(t-1)*m_stage_friction .+ stage_friction_ineq for t=1:T-1]...))

# MathOptInterface problem
prob_nominal_moi = init_MOI_Problem(prob_nominal)
prob_friction_moi = init_MOI_Problem(prob_friction)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.1*rand(model_nominal.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob_nominal)

# Solve nominal problem
@time Z_nominal = solve(prob_nominal_moi,copy(Z0),tol=1.0e-5,c_tol=1.0e-5)
@time Z_friction_nominal = solve(prob_friction_moi,copy(Z0),tol=1.0e-5,c_tol=1.0e-5)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob_nominal)
X_friction_nominal, U_friction_nominal, H_friction_nominal = unpack(Z_friction_nominal,prob_friction)

# Time trajectories
t_nominal = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
end

# Plots results
S_friction_nominal = [U_friction_nominal[t][7] for t=1:T-1]
@assert sum(S_friction_nominal) < 1.0e-4
b_friction_nominal = [U_friction_nominal[t][2] - U_friction_nominal[t][3] for t=1:T-1]

# Control
plt = plot(t_nominal[1:T-1],hcat(U_nominal...)[1:1,:]',color=:purple,width=2.0,
    title="Cartpole",xlabel="time (s)",ylabel="control",label="nominal",
    legend=:topright,linetype=:steppost)
plt = plot!(t_nominal[1:T-1],hcat(U_friction_nominal...)[1:1,:]',color=:orange,width=2.0,
    label="nominal (friction)",linetype=:steppost)

# States
plt = plot(t_nominal,hcat(X_nominal...)[1:4,:]',
    color=:purple,width=2.0,xlabel="time (s)",
    ylabel="state",label=["x (nominal)" "θ (nominal)" "dx (nominal)" "dθ (nominal)"],
    title="Cartpole",legend=:topright)
plt = plot!(t_nominal,hcat(X_friction_nominal...)[1:4,:]',
    color=:orange,width=2.0,
    label=["x (nominal friction)" "θ (nominal friction)" "dx (nominal friction)" "dθ (nominal friction)"],
    )

b_nom = [U_nominal[t][2:3] for t = 1:T-1]
b_friction_nominal = [U_friction_nominal[t][2:3] for t = 1:T-1]
b_friction_nominal = [U_friction_nominal[t][2] - U_friction_nominal[t][3] for t = 1:T-1][20]
(model_friction.mp + model_friction.mc)*model_friction.g*model_friction.μ
plot(hcat(b_nom...)',linetype=:steppost)
plot(hcat(b_friction_nominal...)',linetype=:steppost)

# Sample
N = 2*model.nx

μ_sample = range(0.1,stop=0.3,length=N)
models = [CartpoleFriction(1.0,0.2,0.5,9.81,μ_sample[i],
    nx_friction,nu_friction,nu_policy_friction) for i = 1:N]

β = 1.0
w = 1.0e-3*ones(model_friction.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[1]
    xu_traj_sample[i][1] = x1_sample[1]
end

K = TVLQR_gains(model,X_friction_nominal,U_friction_nominal,H_friction_nominal,Q_lqr,R_lqr,u_policy=(1:1))

prob_sample = init_sample_problem(prob_friction,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    u_policy=(1:1),
    general_objective=true,
    sample_general_constraints=true,
    m_sample_general=prob_friction.m_general*N,
    sample_general_ineq=vcat([(i-1)*prob_friction.m_general .+ (1:prob_friction.m_general) for i = 1:N]...))

prob_sample_moi = init_MOI_Problem(prob_sample)

Ū_friction_nominal = deepcopy(U_friction_nominal)
# for t=1:T-1
#     Ū_nominal[t][2:7] = 0.1*rand(model.nu-1)
# end
Z0_sample = pack(X_friction_nominal,Ū_friction_nominal,H_friction_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),nlp=:SNOPT7)
Z_sample_sol = solve(prob_sample_moi,copy(Z_sample_sol),nlp=:SNOPT7)


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
plt_ctrl = plot!(t_nominal[1:end-1],hcat(U_nominal...)[1:1,:]',color=:purple,
    width=2.0,label="nominal")
plt_ctrl = plot!(t_sample[1:end-1],hcat(U_nom_sample...)[1:1,:]',color=:orange,
    width=2.0,label="nominal (friction)")
display(plt_ctrl)
savefig(plt_ctrl,joinpath(@__DIR__,"results/cartpole_friction_control.png"))

plt_state = plot(title="Cartpole w/ friction state",xlabel="time (s)",
    color=:red,width=2.0)
for i = 1:N
    plt_state = plot!(t_sample,hcat(X_sample[i]...)[1:4,:]',label="")
end
plt_state = plot!(t_sample,hcat(X_nominal...)[1:4,:]',color=:purple,
    width=2.0,label=["nominal" "" "" ""])
plt_state = plot!(t_sample,hcat(X_nom_sample...)[1:4,:]',color=:orange,
    width=2.0,label=["nominal (friction)" "" "" ""])
display(plt_state)
savefig(plt_state,joinpath(@__DIR__,"results/cartpole_friction_state.png"))

S_nominal = [U_nom_sample[t][7] for t=1:T-1]
@assert sum(S_nominal) < 1.0e-4
