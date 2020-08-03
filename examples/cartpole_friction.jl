include("../src/sample_trajectory_optimization.jl")
include("../dynamics/cartpole.jl")
using Plots

model = model_friction

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

# Objective
Q = [t < T ? Diagonal(ones(model.nx)) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal([0.1,0.0,0.0,0.0,0.0,0.0,1000.0]) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T]) # NOTE: there is a discrepancy between paper and DRAKE

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0;1.0;1.0]) : Diagonal(100.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal([0.1,0.0,0.0,0.0,0.0,0.0,0.0]) for t = 1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
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
# t_robust = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
    # t_robust[t] = t_robust[t-1] + H_robust[t-1]
end

# Plots results
S_nominal = [U_nominal[t][7] for t=1:T-1]
b_nominal = [U_nominal[t][2] - U_nominal[t][3] for t=1:T-1]
# Control
plt = plot(t_nominal[1:T-1],hcat(U_nominal...)[1:1,:]',color=:purple,width=2.0,
    title="Cartpole",xlabel="time (s)",ylabel="control",label="nominal",
    legend=:topright,linetype=:steppost)
# plt = plot!(t_robust[1:T-1],Array(hcat(U_robust...))',color=:orange,
#     width=2.0,label="robust",linetype=:steppost)
# savefig(plt,joinpath(@__DIR__,"results/cartpole_control.png"))

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
# plt = plot!(t_robust,hcat(X_robust...)[1,:],
#     color=:orange,width=2.0,label="x (robust)")
# plt = plot!(t_robust,hcat(X_robust...)[2,:],
#     color=:orange,width=2.0,label="θ (robust)")
# plt = plot!(t_robust,hcat(X_robust...)[3,:],
#     color=:orange,width=2.0,label="dx (robust)")
# plt = plot!(t_robust,hcat(X_robust...)[4,:],
#     color=:orange,width=2.0,label="dθ (robust)")
# savefig(plt,joinpath(@__DIR__,"results/cartpole_state.png"))
