include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/quadrotor2D.jl"))
using Plots

# Horizon
T = 51

# Bounds

# ul <= u <= uu
uu = 10.0
ul = 0.0

# hl <= h <= hu
tf0 = 10.0
h0 = tf0/(T-1) # timestep

hu = h0
hl = 0.0

# Initial and final states
x1 = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
xT = [5.0; 1.0; 0.0; 0.0; 0.0; 0.0]

xl_traj = [-Inf*ones(model.nx) for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT

# Objective (minimum time)
Q = [Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal(zeros(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [zeros(model.nx) for t=1:T],[zeros(model.nu) for t=1:T])

# Problem
prob = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Initialization
X0 = linear_interp(x1,xT,T) # linear interpolation for states
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

plot(hcat(X_nominal...)')
plot(hcat(U_nominal...)')

# Samples
Q_lqr = [(t < T ? Diagonal([10.0;10.0;10.0;10.0;10.0;10.0])
	: Diagonal([100.0;100.0;100.0;100.0;100.0;100.0])) for t = 1:T]
R_lqr = [Diagonal(ones(nu)) for t = 1:T-1]
H_lqr = [10.0 for t = 1:T-1]
K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-2*ones(model.nx)
γ = 1.0

α = 1.0e-1
x11 = x1 + α*[1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x12 = x1 + α*[-1.0; 0.0; 0.0; 0.0; 0.0; 0.0]
x13 = x1 + α*[0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
x14 = x1 + α*[0.0; -1.0; 0.0; 0.0; 0.0; 0.0]
x15 = x1 + α*[0.0; 0.0; 1.0; 0.0; 0.0; 0.0]
x16 = x1 + α*[0.0; 0.0; -1.0; 0.0; 0.0; 0.0]
x17 = x1 + α*[0.0; 0.0; 0.0; 1.0; 0.0; 0.0]
x18 = x1 + α*[0.0; 0.0; 0.0; -1.0; 0.0; 0.0]
x19 = x1 + α*[0.0; 0.0; 0.0; 0.0; 1.0; 0.0]
x110 = x1 + α*[0.0; 0.0; 0.0; 0.0; -1.0; 0.0]
x111 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; 1.0]
x112 = x1 + α*[0.0; 0.0; 0.0; 0.0; 0.0; -1.0]

x1_sample = [x11,x12,x13,x14,x15,x16,x17,x18,x19,x110,x111,x112]


xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]
end

prob_sample = init_sample_problem(prob,models,
    Q_lqr,R_lqr,H_lqr,
    β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample)

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
#NOTE: Ipopt finds different solution compared to SNOPT
Z_sample_sol = solve(prob_sample_moi,Z0_sample,nlp=:SNOPT7,time_limit=60*5.0,
	c_tol=1.0e-2,tol=1.0e-2)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
Θ = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]
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
    color=:purple,width=2.0,title="Quadrotor2D",xlabel="time (s)",
    ylabel="control",label="nominal",linelegend=:topleft,
    linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',
    color=:orange,width=2.0,label="sample",linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/quadrotor2D_control.png"))

# States
plt = plot(t_nominal,hcat(X_nominal...)[1,:],
    color=:purple,width=2.0,xlabel="time (s)",ylabel="state",
    label="θ (nominal)",title="Quadrotor 2D",legend=:topleft)
plt = plot!(t_nominal,hcat(X_nominal...)[2,:],color=:purple,width=2.0,label="x (nominal)")
plt = plot!(t_sample,hcat(X_nom_sample...)[1,:],color=:orange,width=2.0,label="θ (sample)")
plt = plot!(t_sample,hcat(X_nom_sample...)[2,:],color=:orange,width=2.0,label="x (sample)")
savefig(plt,joinpath(@__DIR__,"results/quadrotor2D_state.png"))

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
savefig(plt,joinpath(@__DIR__,"results/quadrotor2D_sample_state.png"))

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
savefig(plt,joinpath(@__DIR__,"results/quadrotor2D_sample_control.png"))

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
    xmin=0., ymin=-11, xmax=max(sum(H_nom_sample),sum(H_nominal)), ymax=11,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="state",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"minimum_time_quadrotor2D_state.tikz"), a, include_preamble=false)

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
PGF.save(joinpath(dir,"minimum_time_quadrotor2D_control.tikz"), a, include_preamble=false)

# Simualate
using Distributions
T_sim = 10T

W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-2*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-3*ones(nx)))
w0 = rand(W0,1)


model_sim = model

t_sim_nominal = range(0,stop=H_nominal[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nominal,U_nominal,model_sim,Q_lqr,R_lqr,T_sim,H_nominal[1],
	vec(X_nominal[1]+w0),w,ul=ul,uu=uu)

plt_tvlqr_nom = plot(t_nominal,hcat(X_nominal...)[1:2,:]',legend=:topleft,color=:red,
    label="",
    width=2.0,xlabel="time",title="Quadrotor 2D",ylabel="state")
plt_tvlqr_nom = plot!(t_sim_nominal,hcat(z_tvlqr...)[1:2,:]',color=:purple,
    label="",width=2.0)
savefig(plt_tvlqr_nom,joinpath(@__DIR__,"results/quadrotor2D_tvlqr_nom_sim.png"))

z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_linear_controller(Θ,
    X_nom_sample,U_nom_sample,
    model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],vec(X_nom_sample[1]+w0),w,ul=ul,uu=uu,
	controller=:policy)

plt_sample = plot(t_sample,hcat(X_nom_sample...)[1:2,:]',legend=:bottom,color=:red,
    label=["nominal (sample)" ""],
    width=2.0,xlabel="time",title="Cartpole",ylabel="state")
plt_sample = plot!(t_sim_sample,hcat(z_sample...)[1:2,:]',color=:orange,
    label=["sample" ""],width=2.0,legend=:topleft)
savefig(plt_sample,joinpath(@__DIR__,"results/quadrotor2D_sample_sim.png"))

# cost
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
