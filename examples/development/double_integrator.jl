include("../src/sample_motion_planning.jl")
include("../dynamics/double_integrator.jl")
using Plots

# Horizon
T = 21

tf0 = 2.0
h0 = tf0/(T-1) # time step

# Bounds

# ul <= u <= uu
uu = 10.0
ul = -10.0

# hl <= h <= hu
hu = h0
hl = h0

# Initial and final states
x1 = [1.0; 0.0]
xT = [0.0; 0.0]

# Objective (minimum time)
Q = [Diagonal(ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-1ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [zeros(model.nx) for t=1:T],[zeros(model.nu) for t=1:T])

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;1.0]) : Diagonal([100.0; 100.0]) for t = 1:T]
R_lqr = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]
H_lqr = [0.0 for t=1:T-1]

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Initialization
X0 = linear_interp(x1,xT,T) # linear interpolation for states
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

# TVLQR policy
A = []
B = []
for t = 1:T-1
    x = X_nominal[t]
    u = U_nominal[t]
    h = H_nominal[t]
    x⁺ = X_nominal[t+1]

    fx(z) = discrete_dynamics(model,x⁺,z,u,h,t)
    fu(z) = discrete_dynamics(model,x⁺,x,z,h,t)
    fx⁺(z) = discrete_dynamics(model,z,x,u,h,t)

    A⁺ = ForwardDiff.jacobian(fx⁺,x⁺)
    push!(A,-A⁺\ForwardDiff.jacobian(fx,x))
    push!(B,-A⁺\ForwardDiff.jacobian(fu,u))
end

K = TVLQR(A,B,Q_lqr,R_lqr)

# Samples
N = 2*model.nx
models = [model for i = 1:N]
# K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 1.0e-4*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    disturbance_ctrl=true,α=1000.0)
prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)
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
savefig(plt,joinpath(@__DIR__,"results/double_integrator_control.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ (nominal)",title="Double Integrator",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="x (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="x (sample)")
savefig(plt,joinpath(@__DIR__,"results/double_integrator_state.png"))

# State samples
plt1 = plot(t_sample,hcat(X_nom_sample...)[1,:],color=:red,width=2.0,title="",
    label="");
for i = 1:N
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1,:],label="");
end

plt2 = plot(t_sample,hcat(X_nom_sample...)[2,:],color=:red,width=2.0,label="");
for i = 1:N
    plt2 = plot!(t_sample,hcat(X_sample[i]...)[2,:],label="");
end
plt12 = plot(plt1,plt2,layout=(2,1),title=["x" "ẋ"],xlabel="time (s)")
savefig(plt,joinpath(@__DIR__,"results/double_integrator_sample_state.png"))

# Control samples
plt3 = plot(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],color=:red,width=2.0,
    title="Control",label="",xlabel="time (s)",linetype=:steppost);
for i = 1:N
    plt3 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)[1,:],label="",
        linetype=:steppost);
end
display(plt3)
savefig(plt,joinpath(@__DIR__,"results/double_integrator_sample_control.png"))

# Disturbance controls
plot(hcat(Uw_sample[1]...)',linetype=:steppost,labels="",title="Disturbance controls")
plot!(hcat(Uw_sample[2]...)',linetype=:steppost,labels="")
plot!(hcat(Uw_sample[3]...)',linetype=:steppost,labels="")
plot!(hcat(Uw_sample[4]...)',linetype=:steppost,labels="")
