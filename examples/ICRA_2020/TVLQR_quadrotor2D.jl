include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/quadrotor2D.jl"))
using Plots

nx = model.nx
nu = model.nu

# horizon
T = 201
Δt = 0.05

Δt*(T-1)

# Initial and final states
x1_nom = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
xT_nom = [1.0; 1.0; 0.0; 0.0; 0.0; 0.0]

x_nom_ref = linear_interp(x1_nom,xT_nom,T)

Q_nom = [(t < T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 10.0])
	: Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 10.0])) for t = 1:T]
R_nom = [Diagonal(10.0*ones(model.nu)) for t = 1:T-1]

xl_traj = [-Inf*ones(nx) for t = 1:T]
xu_traj = [Inf*ones(nx) for t = 1:T]

xl_traj[1] = copy(x1_nom)
xu_traj[1] = copy(x1_nom)

xl_traj[T] = copy(xT_nom)
xu_traj[T] = copy(xT_nom)

ul_traj = [zeros(nu) for t = 1:T]
uu_traj = [Inf*ones(nu) for t = 1:T]

u_ref = model.m*model.g/2.0*ones(model.nu)

obj = QuadraticTrackingObjective(
	Q_nom,
	R_nom,
    [xT_nom for t=1:T],[u_ref for t=1:T])

# Problem
prob = init_problem(T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                   	Δt=Δt
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = x_nom_ref # linear interpolation on state
U0 = [copy(u_ref) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom = unpack(Z_nominal,prob)

Plots.plot(hcat(X_nom...)')
Plots.plot(hcat(U_nom...)',linetype=:steppost)

obj_fixed = QuadraticTrackingObjective(
	[Diagonal(zeros(nx)) for t = 1:T],
	[Diagonal(zeros(nu)) for t = 1:T-1],
    [zeros(nx) for t=1:T],[zeros(nu) for t=1:T])

xl_traj = [zeros(nx) for t = 1:T]
xu_traj = [zeros(nx) for t = 1:T]

ul_traj = [zeros(nu) for t = 1:T-1]
uu_traj = [zeros(nu) for t = 1:T-1]

prob_fixed = init_problem(T,model,obj_fixed,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
					Δt=Δt
                    )
# Sample
γ = 1.0
Q_lqr = [(t < T ? Diagonal(γ*[10.0;10.0;10.0;10.0;10.0;10.0])
	: Diagonal(γ*[100.0;100.0;100.0;100.0;100.0;100.0])) for t = 1:T]
R_lqr = [γ*Diagonal(1.0*ones(nu)) for t = 1:T-1]

A_dyn, B_dyn = nominal_jacobians(model,X_nom,U_nom,Δt=Δt)
K = TVLQR_gains(model,X_nom,U_nom,Q_lqr,R_lqr,Δt=Δt)

α = 1.0
# x11 = α*[1.0; 1.0]
# x12 = α*[1.0; -1.0]
# x13 = α*[-1.0; 1.0]
# x14 = α*[-1.0; -1.0]
# x1_sample = [x11,x12,x13,x14]

μ1 = ones(model.nx)
L1 = lt_to_vec(cholesky(Diagonal(α*ones(model.nx))).L)

sample_model = model
β_resample = 1.0
β_con = 1.0
W = [Diagonal(1.0*ones(model.nw)) for t = 1:T-1]

μl_traj_sample = [-Inf*ones(nx) for t = 1:T]
μu_traj_sample = [Inf*ones(nx) for t = 1:T]

μl_traj_sample[1] = copy(μ1)
μu_traj_sample[1] = copy(μ1)

Ll_traj_sample = [-Inf*ones(n_tri(nx)) for t = 1:T]
Lu_traj_sample = [Inf*ones(n_tri(nx)) for t = 1:T]

Ll_traj_sample[1] = copy(L1)
Lu_traj_sample[1] = copy(L1)

# ul_traj_sample = [[-Inf*ones(nu) for t = 1:T-1] for i = 1:N]
# uu_traj_sample = [[Inf*ones(nu) for t = 1:T-1] for i = 1:N]

prob_sample = init_DPO_problem(prob,sample_model,
	Q_lqr,R_lqr,
	μl=μl_traj_sample,
    μu=μu_traj_sample,
    Ll=Ll_traj_sample,
    Lu=Lu_traj_sample,
    # ul=deepcopy(prob.ul),
    # uu=deepcopy(prob.uu),
    # sample_control_constraints=true,
    # sample_state_constraints=false,
    # xl=[-100.0*ones(sample_model.nx) for t = 1:T],
    # xu=[100.0*ones(sample_model.nx) for t = 1:T],
    β_resample=β_resample,β_con=β_con,W=W)

prob_sample_moi = init_MOI_Problem(prob_sample)

# Z0_sample = ones(prob_sample_moi.n)
Z0_sample = pack([rand(nx) for t = 1:T],
	[rand(nu) for t = 1:T-1],
	[rand(nx) for t = 1:T],
	[lt_to_vec(cholesky(Diagonal(rand(nx))).L) for t = 1:T],
	[K[t] for t = 1:T-1],prob_sample)

# Z0_sample = pack([zeros(nx) for t = 1:T],
# 	[zeros(nu) for t = 1:T-1],
# 	[zeros(nx) for t = 1:T],
# 	[lt_to_vec(cholesky(Diagonal(ones(nx))).L) for t = 1:T],
# 	[K[t] for t = 1:T-1],prob_sample)

# linear dynamics
function discrete_dynamics(model::Quadrotor2D,x⁺,x,u,h,w,t)
    x⁺ - A_dyn[t]*x - B_dyn[t]*u - w
end

function discrete_dynamics(model::Quadrotor2D,x,u,h,w,t)
    A_dyn[t]*x + B_dyn[t]*u + w
end

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),
	nlp=:SNOPT7,time_limit=100,tol=1.0e-2,c_tol=1.0e-2)

# Unpack solutions
X_nom_sample, U_nom_sample, μ_sol, L_sol, K_sol, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)


P_sol = [vec_to_lt(L_sol[t])*vec_to_lt(L_sol[t])' for t = 1:T]
Θ_linear = [reshape(K_sol[t],nu,nx) for t = 1:T-1]
policy_error_linear = [norm(vec(Θ_linear[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error (Inf norm) [linear dynamics]:
    $(norm(policy_error_linear,Inf))")

plt = plot(policy_error_linear,xlabel="time step",ylabel="matrix-norm error",yaxis=:log,
    ylims=(1.0e-16,1.0),width=2.0,legend=:bottom,label="")
savefig(plt,joinpath(@__DIR__,"results/TVLQR_quadrotor2D.png"))

# # nonlinear dynamics
# function discrete_dynamics(model::Quadrotor2D,x⁺,x,u,h,t)
# 	midpoint_implicit(model,x⁺,x,u,h)
# end
#
# # Solve
# Z_sample_nonlinear_sol = solve(prob_sample_moi,copy(Z_sample_sol),nlp=:ipopt,
# 	time_limit=60*10,tol=1.0e-2,c_tol=1.0e-2)
#
# # Unpack solutions
# X_nom_sample_nonlinear, U_nom_sample_nonlinear, H_nom_sample_nonlinear, X_sample_nonlinear, U_sample_nonlinear, H_sample_nonlinear = unpack(Z_sample_sol,prob_sample)
#
# Θ_nonlinear = [reshape(Z_sample_nonlinear_sol[prob_sample.idx_K[t]],nu,nx) for t = 1:T-1]
# policy_error_nonlinear = [norm(vec(Θ_nonlinear[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
# println("Policy solution difference (avg.) [nonlinear dynamics]:
#     $(sum(policy_error_nonlinear)/T)")

using PGFPlots
const PGF = PGFPlots

px = PGF.Plots.Linear(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[1,:],
	mark="",style="color = orange, line width = 2pt",legendentry="x")
pz = PGF.Plots.Linear(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[2,:],
	mark="",style="color = purple, line width = 2pt",legendentry="y")
pθ = PGF.Plots.Linear(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[3,:],
	mark="",style="color = cyan, line width = 2pt",legendentry="theta")

a = Axis([px;pz;pθ],
    xmin=0.0, ymin=-0.1, xmax=5.0, ymax=1.05,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="configuration",
	xlabel="time",
	legendStyle="{at={(0.01,0.99)},anchor=north west}"
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"quadrotor2D_traj.tikz"), a, include_preamble=false)

p = PGF.Plots.Linear(range(1,stop=T-1,length=T-1),policy_error_linear,mark="",style="color=black, line width = 2pt")

a = Axis(p,
    xmin=1., ymin=1.0e-16, xmax=T-1, ymax=1.0,
    axisEqualImage=false,
    hideAxis=false,
	ylabel="matrix-norm error",
	xlabel="time step",
	ymode="log",
	)

# Save to tikz format
dir = joinpath(@__DIR__,"results")
PGF.save(joinpath(dir,"TVLQR_quadrotor2D.tikz"), a, include_preamble=false)


plot(range(0,stop=Δt*(T-1),length=T),hcat(X_nom...)[1:3,:]')

# Simulate policy
using Distributions
include(joinpath(pwd(),"dynamics/quadrotor2D.jl"))

model_sim = model
T_sim = 10*T

W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-5*ones(nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(X_nom[1]) + w0)

t_nom = range(0,stop=Δt*T,length=T)
t_sim = range(0,stop=Δt*T,length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,Δt,z0_sim,w,_norm=2)
#
z_linear, u_linear, J_linear, Jx_linear, Ju_linear = simulate_linear_controller(Θ_linear,
    X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,Δt,z0_sim,w,_norm=2)
#
# z_nonlin, u_nonlin, J_nonlin, Jx_nonlin, Ju_nonlin = simulate_linear_controller(Θ_nonlinear,
#     X_nom,U_nom,model_sim,Q_lqr,R_lqr,T_sim,Δt,z0_sim,w,_norm=2)
#
plt_x = plot(t_nom,hcat(X_nom...)[1:6,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Quadrotor 2D",ylabel="state")
plt_x = plot!(t_sim,hcat(z_tvlqr...)[1:6,:]',color=:purple,label="",
    width=2.0)
plt_x = plot!(t_sim,hcat(z_linear...)[1:6,:]',linetype=:steppost,color=:cyan,
    label="",width=2.0)
# plt_x = plot!(t_sim,hcat(z_nonlin...)[1:6,:]',linetype=:steppost,color=:orange,
#     label="",width=2.0)
#
# plt_u = plot(t_nom[1:T-1],hcat(U_nom...)[1:2,:]',legend=:topright,color=:red,
#     label="",width=2.0,xlabel="time (s)",
#     title="Quadrotor 2D",ylabel="control",linetype=:steppost)
# plt_u = plot!(t_sim[1:T_sim-1],hcat(u_tvlqr...)[1:1,:]',color=:purple,
#     label="tvlqr",width=2.0)
# plt_u = plot!(t_sim[1:T_sim-1],hcat(u_linear...)[1:1,:]',color=:cyan,
#     label="linear",width=2.0)
# plt_u = plot!(t_sim[1:T_sim-1],hcat(u_nonlin...)[1:1,:]',color=:orange,
#     label="nonlinear",width=2.0)
#
# objective value
J_tvlqr
J_linear
J_nonlin

# state tracking
Jx_tvlqr
Jx_linear
Jx_nonlin

# control tracking
Ju_tvlqr
Ju_linear
Ju_nonlin
