include(joinpath(pwd(),"src/direct_policy_optimization.jl"))
include(joinpath(pwd(),"dynamics/rocket.jl"))
using Plots

# Model
model = model
nx = model.nx
nu = model.nu

# Horizon
T = 51

# Initial and final states
x1 = [5.0; model.l2+10.0; -5*pi/180.0; -1.0; -1.0; -0.5*pi/180.0]
xT = [0.0; model.l2; 0.0; 0.0; 0.0; 0.0]

# Bounds

# xl <= x <= xl
xl = -Inf*ones(model.nx)
xl[2] = model.l2
xu = Inf*ones(model.nx)

xl_traj = [xl for t = 1:T]
xu_traj = [xu for t = 1:T]

xl_traj[1] = x1
xu_traj[1] = x1

xl_traj[T] = xT
xu_traj[T] = xT
xl_traj[T][1] = -0.5
xu_traj[T][1] = 0.5


# ul <= u <= uu
uu = [25.0;5.0;10*pi/180.0]
ul = [0.0;-5.0;-10*pi/180.0]

tf0 = 10.0
h0 = tf0/(T-1)
hu = 10*h0
hl = 0*h0

# Objective
Q = [(t != T ? Diagonal([1.0*ones(3);1.0*ones(3)])
    : Diagonal([100.0*ones(3);100.0*ones(3)])) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T])

# Problem
prob = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul for t=1:T-1],
                    uu=[uu for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [1.0e-1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

#NOTE: may need to run examples multiple times to get good trajectories
# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

@show sum(H_nom)
x_pos = [X_nom[t][1] for t = 1:T]
z_pos = [X_nom[t][2] for t = 1:T]

plot(x_pos,z_pos,xlabel="x",ylabel="z",title="Rocket trajectory",
    aspect_ratio=:equal)

plot(x_pos)
plot(z_pos)
plot(hcat(U_nom...)',linetype=:steppost)

# TVLQR policy
Q_lqr = [(t < T ? Diagonal([10.0*ones(3);10.0*ones(3)])
   : Diagonal([100.0*ones(3);100.0*ones(3)])) for t = 1:T]
R_lqr = [Diagonal(1.0*ones(model.nu)) for t = 1:T-1]
H_lqr = [10.0 for t = 1:T-1]

K = TVLQR_gains(model,X_nom,U_nom,H_nom,Q_lqr,R_lqr)

N = 2*model.nx
models = [model for i = 1:N]
β = 1.0
w = 1.0e-2*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

for i = 1:N
    xl_traj_sample[i][1] = x1_sample[1]
    xu_traj_sample[i][1] = x1_sample[1]

    xl_traj_sample[i][T] = xT
    xu_traj_sample[i][T] = xT
    xl_traj_sample[i][T][1] = -0.5
    xu_traj_sample[i][T][1] = 0.5
end

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    n_policy=model.nu)

prob_sample_moi = init_MOI_Problem(prob_sample)

# Z0_sample = pack(X0,U0,h0,K0,prob_sample)
Z0_sample = pack(X_nom,U_nom,h0,K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,copy(Z0_sample),max_iter=500,nlp=:SNOPT7,time_limit=60*20)
Z_sample_sol = solve(prob_sample_moi,copy(Z_sample_sol),max_iter=500,nlp=:SNOPT7,time_limit=60*20)

# Unpack solutions
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
Θ = [reshape(Z_sample_sol[prob_sample.idx_K[t]],model.nu,model.nx) for t = 1:T-1]
# Time trajectories
t_nominal = zeros(T)
t_sample = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nom[t-1]
    t_sample[t] = t_sample[t-1] + H_nom_sample[t-1]
end

display("time (nominal): $(sum(H_nom))s")
display("time (sample): $(sum(H_nom_sample))s")

# Plots results
plot(x_pos,z_pos,xlabel="x",ylabel="z",title="Rocket trajectory",
    aspect_ratio=:equal,color=:purple,width=2.0,label="nominal (tf=$(round(sum(H_nom),digits=3))s)")
x_sample_pos = [X_nom_sample[t][1] for t = 1:T]
z_sample_pos = [X_nom_sample[t][2] for t = 1:T]
plt = plot!(x_sample_pos,z_sample_pos,aspect_ratio=:equal,width=2.0,label="sample  (tf=$(round(sum(H_nom_sample),digits=3))s)",color=:orange,legend=:bottomright)
savefig(plt,joinpath(@__DIR__,"results/rocket_trajectory.png"))

# Control
plt = plot(t_nominal[1:T-1],Array(hcat(U_nom...))',width=2.0,
    title="Rocket",xlabel="time (s)",ylabel="control",
    label=["FE (nominal)" "FT (nominal)" "φ (nominal)"],color=:purple,
    legend=:top,linetype=:steppost)
plt = plot!(t_sample[1:T-1],Array(hcat(U_nom_sample...))',color=:orange,
    width=2.0,label=["FE (sample nominal)" "FT (sample nominal)" "φ (sample nominal)"],linetype=:steppost)
savefig(plt,joinpath(@__DIR__,"results/rocket_control.png"))

# Samples

# State samples
plt1 = plot(title="Sample states",legend=:topright,xlabel="time (s)");
for i = 1:N
    plt1 = plot!(t_sample,hcat(X_sample[i]...)[1:3,:]',label="");
end
plt1 = plot!(t_sample,hcat(X_nom_sample...)[1:3,:]',color=:red,width=2.0,
    label=label=["x" "z" "θ"])
display(plt1)
savefig(plt1,joinpath(@__DIR__,"results/rocket_sample_states.png"))

# Control samples
plt2 = plot(title="Sample controls",xlabel="time (s)",legend=:topleft);
for i = 1:N
    plt2 = plot!(t_sample[1:end-1],hcat(U_sample[i]...)',label="",
        linetype=:steppost);
end
plt2 = plot!(t_sample[1:end-1],hcat(U_nom_sample...)',color=:red,width=2.0,
    label=["FE (sample nominal)" "FT (sample nominal)" "φ (sample nominal)"],linetype=:steppost)
display(plt2)
savefig(plt2,joinpath(@__DIR__,"results/rocket_sample_controls.png"))

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

l1 = Cylinder(Point3f0(0,0,-model.l2),Point3f0(0,0,model.l1),convert(Float32,0.1))
setobject!(vis["rocket"],l1,MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

anim = MeshCat.Animation(convert(Int,floor(1/H_nom[1])))
H_nom[1]
for t = 1:T
    MeshCat.atframe(anim,t) do
        settransform!(vis["rocket"], compose(Translation(X_nom[t][1],0.0,X_nom[t][2]),LinearMap(RotY(X_nom[t][3]))))
    end
end
MeshCat.setanimation!(vis,anim)
# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))

# Simulate policy
using Distributions
model_sim = model
T_sim = 10*T

W = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-3*ones(nx)))
w = rand(W,T_sim)

W0 = Distributions.MvNormal(zeros(nx),Diagonal(1.0e-3*ones(nx)))
w0 = rand(W0,1)

z0_sim = vec(copy(X_nom[1]) + w0)

t_nom = range(0,stop=sum(H_nom),length=T)
t_sim_nom = range(0,stop=sum(H_nom),length=T_sim)
t_sim_sample = range(0,stop=sum(H_nom_sample),length=T_sim)

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,
    X_nom,U_nom,model_sim,Q,R,T_sim,H_nom[1],z0_sim,w,_norm=2)

z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_linear_controller(Θ,
    X_nom_sample,U_nom_sample,model_sim,Q,R,T_sim,H_nom_sample[1],z0_sim,w,_norm=2)

plt_x = plot(t_nom,hcat(X_nom...)[1:nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_nom,hcat(z_tvlqr...)[1:nx,:]',color=:purple,label="tvlqr",
    width=2.0)

plt_x = plot(t_sample,hcat(X_nom_sample...)[1:nx,:]',legend=:topright,color=:red,
    label="",width=2.0,xlabel="time (s)",
    title="Rocket",ylabel="state")
plt_x = plot!(t_sim_sample,hcat(z_sample...)[1:nx,:]',linetype=:steppost,color=:orange,
    label="",width=2.0)

# plt_u = plot(t_nom[1:T-1],hcat(u_nom...)[1:1,:]',legend=:topright,color=:red,
#     label=["nominal"],width=2.0,xlabel="time (s)",
#     title="Pendulum",ylabel="control",linetype=:steppost)
# plt_u = plot!(t_sim[1:T_sim-1],hcat(u_tvlqr...)[1:1,:]',color=:purple,label="tvlqr",
#     width=2.0)
# plt_u = plot!(t_sim[1:T_sim-1],hcat(u_nonlin...)[1:1,:]',color=:orange,label="nonlinear",
#     width=2.0)

# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
