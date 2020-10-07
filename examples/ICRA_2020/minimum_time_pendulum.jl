include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/pendulum.jl"))
using Plots

# Model
model = model_ft

# Horizon
T = 51

# Bounds

# ul <= u <= uu
uu = 3.0
ul = -3.0

# hl <= h <= hu
tf0 = 2.0
h0 = tf0/(T-1) # timestep

hu = h0
hl = 0.0

# Initial and final states
x1 = [0.0; 0.0]
xT = [π; 0.0]

xl_traj = [-Inf*ones(model.nx) for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT

# Objective
# Q = [Diagonal(zeros(model.nx)) for t = 1:T]
# R = [Diagonal(zeros(model.nu)) for t = 1:T-1]
# track_obj = QuadraticTrackingObjective(Q,R,
#     [zeros(model.nx) for t=1:T],[zeros(model.nu) for t=1:T])
c = 1.0
ft_obj = FreeTimeObjective(c)

# obj = FreeTimeTrackingObjective(track_obj)

# TVLQR cost
Q_lqr = [t < T ? Diagonal([10.0;10.0]) : Diagonal([100.0; 100.0]) for t = 1:T]
R_lqr = [Diagonal([0.1;10.0]) for t = 1:T-1]

# Problem
prob = init_problem(T,model,ft_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[[ul;hl] for t=1:T-1],
                    uu=[[uu;hu] for t=1:T-1],
					free_time=true,
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Initialization
X0 = linear_interp(x1,xT,T) # linear interpolation for states
U0 = [[0.0;h0] for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)

# Unpack solutions
X_nominal, U_nominal = unpack(Z_nominal,prob)

plot(hcat(U_nominal...)[1,:])
sum(hcat(U_nominal...)[2,:])

# Samples
N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 2.0e-3*ones(model.nx)
γ = 1.0

α = 1.0e-3
x11 = α*[1.0; 0.0]
x12 = α*[-1.0; 0.0]
x13 = α*[0.0; 1.0]
x14 = α*[0.0; -1.0]
x1_sample = resample([x11,x12,x13,x14],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]
end

K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

prob_sample = init_sample_problem(prob,models,
    Q_lqr,R_lqr,H_lqr,
    β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
#NOTE: Ipopt finds different solution compared to SNOPT
Z_sample_sol = solve(prob_sample_moi,Z0_sample,nlp=:SNOPT7,time_limit=30)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

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
    color=:purple,width=2.0,title="Pendulum",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/pendulum_control_noise.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ (nominal)",title="Pendulum",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="x (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="x (sample)")
savefig(plt,joinpath(@__DIR__,"results/pendulum_state_noise.png"))

# State samples
plt1 = plot(t_sample,hcat(X_nom_sample...)[1,:],color=:red,width=2.0,title="",
    label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1,:],label="");
end

plt2 = plot(t_sample,hcat(X_nom_sample...)[2,:],color=:red,width=2.0,label="");
for i = 1:N
    t_sample = zeros(T)
    for t = 2:T
        t_sample[t] = t_sample[t-1] + H_sample[i][t-1]
    end
    plt2 = plot!(t_sample,hcat(X_sample[i]...)[2,:],label="");
end
plt12 = plot(plt1,plt2,layout=(2,1),title=["θ" "x"],xlabel="time (s)")
savefig(plt,joinpath(@__DIR__,"results/pendulum_sample_state.png"))

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
savefig(plt,joinpath(@__DIR__,"results/pendulum_sample_control.png"))

using PGFPlots
const PGF = PGFPlots

# nominal trajectory
psx_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[1,:],mark="",
	style="color=purple, line width=3pt",legendentry="pos. (TO)")
psθ_nom = PGF.Plots.Linear(t_nominal,hcat(X_nominal...)[2,:],mark="",
	style="color=purple, line width=3pt, densely dashed",legendentry="ang. (TO)")

# DPO trajectory
psx_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[1,:],mark="",
	style="color=orange, line width=3pt",legendentry="pos. (DPO)")
psθ_dpo = PGF.Plots.Linear(t_sample,hcat(X_nom_sample...)[2,:],mark="",
	style="color=orange, line width=3pt, densely dashed",legendentry="ang. (DPO)")

a = Axis([psx_nom;psθ_nom;psx_dpo;psθ_dpo],
    xmin=0., ymin=-3, xmax=max(sum(H_nom_sample),sum(H_nominal)), ymax=7.0,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"minimum_time_pendulum_state.tikz"), a, include_preamble=false)

# nominal trajectory
psu_nom = PGF.Plots.Linear(t_nominal[1:end-1],hcat(U_nominal...)[1,:],mark="",
	style="const plot,color=purple, line width=3pt",legendentry="TO")

# DPO trajectory
psu_dpo = PGF.Plots.Linear(t_sample[1:end-1],hcat(U_nom_sample...)[1,:],mark="",
	style="const plot, color=orange, line width=3pt",legendentry="DPO")

a = Axis([psu_nom;psu_dpo],
    xmin=0., ymin=-3.1, xmax=max(sum(H_nom_sample[1:end-1]),sum(H_nominal[1:end-1])), ymax=3.1,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="control",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"minimum_time_pendulum_control.tikz"), a, include_preamble=false)