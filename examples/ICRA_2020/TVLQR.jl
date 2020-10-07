include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/quadrotor2D.jl"))
using Plots

nx = model.nx
nu = model.nu

# horizon
T = 11
Δt = 0.1

Δt*(T-1)

# Initial and final states
x1_nom = [0.0; 1.0; 0.0; 0.0; 0.0; 0.0]
xT_nom = [0.0; 2.0; 0.0; 0.0; 0.0; 0.0]

x_nom_ref = linear_interp(x1_nom,xT_nom,T)

Q_nom = [(t < T ? Diagonal(1.0*ones(model.nx))
	: Diagonal(1.0*ones(model.nx))) for t = 1:T]
R_nom = [Diagonal(10.0*ones(model.nu)) for t = 1:T-1]

xl_traj = [-Inf*ones(nx) for t = 1:T]
xu_traj = [Inf*ones(nx) for t = 1:T]

xl_traj[1] = x1_nom
xu_traj[1] = x1_nom

xl_traj[T] = xT_nom
xu_traj[T] = xT_nom

ul_traj = [-Inf*ones(nu) for t = 1:T]
uu_traj = [Inf*ones(nu) for t = 1:T]

obj = QuadraticTrackingObjective(
	Q_nom,
	R_nom,
    [zeros(model.nx) for t=1:T],[zeros(model.nu) for t=1:T])

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
U0 = [zeros(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom = unpack(Z_nominal,prob)

Plots.plot(hcat(X_nom...)')
Plots.plot(hcat(U_nom...)',linetype=:steppost)

A = []
B = []

for t = 1:T-1
	x = X_nom[t]
	u = U_nom[t]
	fx(z) = midpoint(model,z,u,Δt,zeros(model.nw))
	fu(z) = midpoint(model,x,z,Δt,zeros(model.nw))

	push!(A,ForwardDiff.jacobian(fx,x))
	push!(B,ForwardDiff.jacobian(fu,u))
end


obj = QuadraticTrackingObjective(
	[Diagonal(zeros(nx)) for t = 1:T],
	[Diagonal(zeros(nu)) for t = 1:T-1],
    [zeros(nx) for t=1:T],[zeros(nu) for t=1:T])

# Problem
prob = init_problem(T,model,obj,
                    xl=[zeros(nx) for t = 1:T],
                    xu=[zeros(nx) for t = 1:T],
                    ul=[zeros(nu) for t = 1:T-1],
                    uu=[zeros(nu) for t = 1:T-1],
					Δt=Δt
                    )


# Sample
Q_lqr = [Diagonal(ones(nx)) for t = 1:T]
R_lqr = [Diagonal(ones(nu)) for t = 1:T-1]

K = TVLQR(A,B,Q_lqr,R_lqr)
A
α = 1.0e-1
# x11 = α*[1.0; 1.0]
# x12 = α*[1.0; -1.0]
# x13 = α*[-1.0; 1.0]
# x14 = α*[-1.0; -1.0]
# x1_sample = [x11,x12,x13,x14]

μ1 = zeros(model.nx)
L1 = lt_to_vec(cholesky(Diagonal(α*ones(model.nx))).L)

sample_model = model
β_resample = 1.0
β_con = 1.0
W = [Diagonal(1.0e-1*ones(model.nw)) for t = 1:T-1]

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
	[rand(nu*nx) for t = 1:T-1],prob_sample)

function discrete_dynamics(model::Quadrotor2D,x,u,h,w,t)
    A[t]*x + B[t]*u + w
end

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),
	nlp=:SNOPT7,time_limit=120,tol=1.0e-3,c_tol=1.0e-3)

# Unpack solutions
X_nom_sample, U_nom_sample, μ_sol, L_sol, K_sol, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)
P_sol = [vec_to_lt(L_sol[t])*vec_to_lt(L_sol[t])' for t = 1:T]

Θ = [reshape(K_sol[t],nu,nx) for t = 1:T-1]
policy_error = [norm(vec(Θ[t]-K[t]))/norm(vec(K[t])) for t = 1:T-1]
println("Policy solution error (Inf norm): $(norm(policy_error,Inf))")

using Plots
plt = plot(policy_error,xlabel="time step",ylims=(1.0e-16,1.0),yaxis=:log,
    width=2.0,legend=:bottom,ylabel="matrix-norm error",label="")
savefig(plt,joinpath(@__DIR__,"results/LQR_double_integrator.png"))


plt1 = plot(title="Sample states",legend=:bottom,xlabel="time (s)");
for i = 1:prob_sample.N_sample_con
    # t_sample = zeros(T)
    # for t = 2:T
    #     t_sample[t] = t_sample[t-1] + Δt
    # end
    plt1 = plot!(hcat(X_sample[i]...)',label="");
end
plt1 = plot!(hcat(X_nom_sample...)',color=:red,width=2.0,
    label=["nominal" "" ""])
display(plt1)

# Control samples
plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:bottom);
for i = 1:prob_sample.N_sample_con
    # t_sample = zeros(T)
    # for t = 2:T
    #     t_sample[t] = t_sample[t-1] + Δt
    # end
    plt2 = plot!(hcat(U_sample[i]...)',label="",
        linetype=:steppost);
end
plt2 = plot!(hcat(U_nom_sample...)',color=:red,width=2.0,
    label=["nominal" ""],linetype=:steppost)
display(plt2)

using PGFPlots
const PGF = PGFPlots

p = PGF.Plots.Linear(range(1,stop=T-1,length=T-1),policy_error,mark="",style="color=black, very thick")

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
PGF.save(joinpath(dir,"LQR_double_integrator.tikz"), a, include_preamble=false)
