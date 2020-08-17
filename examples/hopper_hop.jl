include("../src/sample_trajectory_optimization.jl")
include("../dynamics/hopper.jl")
include("../src/loop_up.jl")
using Plots

# Horizon
T = 51
Tm = convert(Int,(T-3)/2 + 3)

tf = 0.35
model.Δt = tf/(T-2)

zh = 0.1
# Initial and final states
x1 = [0., model.r+zh, 0.75*model.r, 0., 0.]
xM = [0., 0.75*model.r, 0.75*model.r, 0., 0.]

# Bounds
# xl <= x <= xu
xu_traj = [model.qU for t=1:T]
xl_traj = [model.qL for t=1:T]

xu_traj[2] = [x1[1:2]; Inf*ones(3)]
# xu_traj[Tm] = xM
# xu_traj[T] = xT

xl_traj[2] = [x1[1:2]; -Inf*ones(3)]
# xl_traj[Tm] = xM
# xl_traj[T] = xT
# ul <= u <= uu
uu = Inf*ones(model.nu)
uu[model.idx_u] .= 100.0
ul = zeros(model.nu)
ul[model.idx_u] .= -100.0

# h = h0 (fixed timestep)
hu = model.Δt
hl = model.Δt

# Objective
Q = [t < Tm ? Diagonal([1.0,1.0,1.0,0.1,0.1]) : Diagonal(5.0*ones(model.nx)) for t = 1:T]
R = [Diagonal([1.0e-1,1.0e-3]) for t = 1:T-2]
c = 0.0



obj = QuadraticTrackingObjective(Q,R,c,
    [x1 for t=1:T],[zeros(model.nu_ctrl) for t=1:T-2])
model.α = 100.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul for t = 1:T-2],
                    uu=[uu for t = 1:T-2],
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2],
					general_constraints=true,
					m_general=model.nx+model.nx,
					general_ineq=(1:0),
					contact_sequence=true,
					T_contact_sequence=[Tm]
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T)
U0 = [0.001*rand(model.nu) for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,model.Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0))
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)
plot(hcat(U_nom...)[model.idx_u,:]',linetype=:steppost)
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]

x_nom = [X_nom[t][1] for t = 1:T]
y_nom = [X_nom[t][2] for t = 1:T]
z_nom = [X_nom[t][3] for t = 1:T]

t_nom = zeros(T-2)
for t = 2:T-2
    t_nom[t] = t_nom[t-1] + H_nom[t-1]
end

@assert norm(s_nom,Inf) < 1.0e-5
@assert norm(ϕ_func(model,X_nom[Tm]) - ϕ_func(model,xM)) < 1.0e-5
@assert norm(X_nom[3][2:end] - X_nom[T][2:end]) < 1.0e-5
λ = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
plot(t_nom,λ,linetype=:steppost)
plot(t_nom,[ϕ_func(model,X_nom[t])[1] for t = 3:T],linetype=:steppost)
sum(H_nom)
using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes
using LinearAlgebra
using MeshCat
using MeshIO
using Rotations

vis = Visualizer()
open(vis)
visualize!(vis,model,X_nom)

# samples
Q_lqr = [t < T ? Diagonal([10.0;10.0;10.0;1.0;1.0]) : Diagonal([10.0; 10.0; 10.0;1.0;1.0]) for t = 1:T]
R_lqr = [Diagonal([1.0; 1.0e-1]) for t = 1:T-2]
H_lqr = [0.0 for t = 1:T-1]

# Samples
N = 2*model.nx
models = [model for i =1:N]
K0 = [rand(model.nu_ctrl*(model.nx + 2*model.nc)) for t = 1:T-2]
β = 1.0
w = 1.0e-8*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
	xl_traj_sample[i][2] = x1_sample[i]
	xu_traj_sample[i][2] = x1_sample[i]
end

ul_traj_sample = [[ul for t = 1:T-2] for i = 1:N]
uu_traj_sample = [[uu for t = 1:T-2] for i = 1:N]

hl_traj_sample = [[hl for t = 1:T-2] for i = 1:N]
hu_traj_sample = [[hu for t = 1:T-2] for i = 1:N]

function policy(model::Hopper,K,x1,x2,x3,ū,h,x1_nom,x2_nom,x3_nom,u_nom,ū_nom,h_nom)
	λ = ū[(1:model.nc)]
	λ_nom = ū_nom[(1:model.nc)]
	u_nom - reshape(K,model.nu_ctrl,model.nx + 2*model.nc)*[(x3 - x3_nom);
																   ϕ_func(model,x3) - ϕ_func(model,x3_nom);
																   λ - λ_nom]
end

prob_sample = init_sample_problem(prob,models,x1_sample,
    Q_lqr,R_lqr,H_lqr,
	u_policy=model.idx_u,
	nK = model.nu_ctrl*(model.nx + 2*model.nc),
    β=β,w=w,γ=γ,
    disturbance_ctrl=true,
    α=1.0e-3,
	ul=ul_traj_sample,
	uu=uu_traj_sample,
	xl=xl_traj_sample,
	xu=xu_traj_sample,
	hl=hl_traj_sample,
	hu=hu_traj_sample,
    sample_general_constraints=true,
    m_sample_general=N*(model.nx + model.nx),
    sample_general_ineq=(1:0),
	general_objective=true,
	sample_contact_sequence=true,
	T_sample_contact_sequence=[[Tm-5],[Tm-4],[Tm-3],[Tm-2],[Tm-1],[Tm+1],[Tm+2],[Tm+3],[Tm+4],[Tm+5]])

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nom,U_nom,H_nom[1],K0,prob_sample)

# Solve
#NOTE: run multiple times to get good solution
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=1000)
Z_sample_sol = solve(prob_sample_moi,Z_sample_sol)

X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)

s = [U_nom_sample[t][model.idx_s] for t = 1:T-2]
@assert norm(s,Inf) < 1.0e-5

t_sample = zeros(T-2)
for t = 2:T-2
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

pltz = plot(label="",xlabel="time (s)",ylabel="z",title="Hopper",
	legend=:topright)

for i = 1:N
	z_sample = [X_sample[i][t][3] for t = 1:T]
	pltz = plot!(t_sample,z_sample[3:T],label="")
end
pltz = plot!(t_nom,z_nom[3:T],color=:purple,label="nominal",width=2.0)
z_nom_sample =  [X_nom_sample[t][3] for t = 1:T]
pltz = plot!(t_sample,z_nom_sample[3:T],color=:orange,label="sample nominal",width=2.0,legend=:bottomright)
display(pltz)
savefig(pltz,joinpath(@__DIR__,"results/hopper_z_T$T.png"))

pltx = plot(label="",xlabel="time (s)",ylabel="x",title="Hopper",
	legend=:bottomright)
t_span = range(0,stop=model.Δt*(T-1),length=T)
for i = 1:N
	x_sample = [X_sample[i][t][1] for t = 1:T]
	pltx = plot!(t_sample,x_sample[3:T],label="")
end
pltx = plot!(t_nom,x_nom[3:T],color=:purple,label="nominal",width=2.0)
x_nom_sample =  [X_nom_sample[t][1] for t = 1:T]
pltx = plot!(t_sample,x_nom_sample[3:T],color=:orange,label="sample nominal",width=2.0)
display(pltx)
savefig(pltx,joinpath(@__DIR__,"results/hopper_x_T$T.png"))

plt_sdf = plot(label="",xlabel="time (s)",ylabel="sdf",title="Hopper",
	legend=:bottomright)
for i = 1:N
	sdf_sample = [ϕ_func(model,X_sample[i][t])[1] for t = 1:T]
	plt_sdf = plot!(t_sample,sdf_sample[3:T],label="")
end
plt_sdf = plot!(t_nom,[ϕ_func(model,X_nom[t])[1] for t = 3:T],color=:purple,label="nominal",width=2.0)
plt_sdf = plot!(t_sample,[ϕ_func(model,X_nom_sample[t])[1] for t = 3:T],color=:orange,label="sample nominal",width=2.0)
display(plt_sdf)
savefig(plt_sdf,joinpath(@__DIR__,"results/hopper_sdf_T$T.png"))

plt_ctrl = plot(label="",xlabel="time (s)",ylabel="ctrl",title="Hopper",
	legend=:bottomright)
for i = 1:N
	plt_ctrl = plot!(t_sample,hcat(U_sample[i]...)[1:model.nu_ctrl,:]',label="")
end
plt_ctrl = plot!(t_nom,hcat(U_nom...)[1:model.nu_ctrl,:]',color=:purple,label="nominal",width=2.0)
plt_ctrl = plot!(t_sample,hcat(U_nom_sample...)[1:model.nu_ctrl,:]',color=:orange,label="sample nominal",width=2.0)
display(plt_ctrl)
savefig(plt_ctrl,joinpath(@__DIR__,"results/hopper_ctrl_T$T.png"))

visualize!(vis,model,X_nom_sample)
