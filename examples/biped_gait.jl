include("../src/sample_trajectory_optimization.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 15
Tm = -1
model.Tm = Tm

# Initial and final states
q_init = [2.44917526290273,2.78819438807838,0.812693850088907,0.951793806080012,1.49974183648719e-13]
v_init = [0.0328917694318260,0.656277832705193,0.0441573173000750,1.03766701983449,1.39626340159558]
q_final = [2.84353387648829,2.48652580597252,0.751072212241267,0.645830978766432, 0.0612754113212848]
v_final = [2.47750069399969,3.99102008940145,-1.91724136219709,-3.95094757056324,0.0492401787458546]

x1 = [q_init;v_init]
# xT = [q_final;v_final]
xT = Δ(x1)

# Bounds

# xl <= x <= xu
# xl_traj = [t != T ? -Inf*ones(model.nx) : xT for t = 1:T]
# xu_traj = [t != T ? Inf*ones(model.nx) : xT for t = 1:T]

# ul <= u <= uu
uu = 20.0
ul = -20.0

tf0 = 0.36915
h0 = tf0/(T-1)
hu = 2.0*h0
hl = 0.0*h0

function discrete_dynamics(model::Biped,x⁺,x,u,h,t)
    if t == model.Tm
        rk3_implicit(model,x⁺,Δ(x),u,h)
    else
        return rk3_implicit(model,x⁺,x,u,h)
    end
end

# Objective
qq = [0,0,0,0,1.0,0,0,0,0,1.0]
Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1.0,0.1,[t for t = 1:T-1 if (t != Tm-1 || t != 1)])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true)#,
                    # stage_constraints=false,
                    # m_stage=[t==Tm ? 0 : m_stage for t=1:T-1]
                    # )

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

Q_nominal = [X_nominal[t][1:5] for t = 1:T]

foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=1:T]
foot_y = [foot_traj[t][2] for t=1:T]

plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)


# start in mid trajectory
T = 15
Tm = 8
model.Tm = Tm
x1 = X_nominal[Tm]

tf0 = 0.5*0.36915
h0 = tf0/(T-1)
hu = 2.0*h0
hl = 0.0*h0

xl_traj = [-Inf*ones(model.nx) for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

xl_traj[1][1:5] = x1[1:5]
xu_traj[1][1:5] = x1[1:5]

xl_traj[Tm][1:5] = xT[1:5]
xu_traj[Tm][1:5] = xT[1:5]

xl_traj[T][1:5] = x1[1:5]
xu_traj[T][1:5] = x1[1:5]

Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1.0e-1,0.1,[t for t = 1:T-1 if (t != Tm)])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
include("../src/loop.jl")
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    initial_constraint=false,
                    goal_constraint=false,
                    general_constraints=true,
                    m_general=model.nx,
                    general_ineq=(1:0))#,
                    # stage_constraints=false,
                    # m_stage=[t==Tm ? 0 : m_stage for t=1:T-1]
                    # )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = [linear_interp(x1,xT,Tm)...,linear_interp(x1,xT,Tm)...] # linear interpolation on state
# X0 = linear_interp(x1,xT,T) # linear interpolation on state

U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

@assert norm(X_nominal[1][1:5] - X_nominal[T][1:5]) < 1.0e-5
@assert norm(X_nominal[Tm][1:5] - xT[1:5]) < 1.0e-5

Q_nominal = [X_nominal[t][1:5] for t = 1:T]

foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=(1:Tm)]
foot_y = [foot_traj[t][2] for t=(1:Tm)]
plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)
@show foot_x[1]
@show foot_x[end]
foot_x = [foot_traj[t][1] for t=(Tm+1):T]
foot_y = [foot_traj[t][2] for t=(Tm+1):T]
@show foot_x[end]

kinematics(model,xT[1:5])
plt_ft_nom = plot!(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:blue)

foot_x = [foot_traj[t][1] for t=(1:T)]
foot_y = [foot_traj[t][2] for t=(1:T)]
plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)


# TVLQR policy
Q_lqr = [t < T ? Diagonal(1.0*ones(model.nx)) : Diagonal(1.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]
H_lqr = [1.0 for t = 1:T-1]

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
w = 1.0e-1*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

prob_sample = init_sample_problem(prob,models,x1_sample,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    disturbance_ctrl=true,α=1.0e-3,
    sample_initial_constraint=false,
    sample_general_constraints=true,
    m_sample_general=N*model.nx,
    sample_general_ineq=(1:0))

for i = 1:N
    prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[1]] = [x1_sample[i][1:5];Inf*ones(5)]

    prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[Tm]] = [xT[1:5];-Inf*ones(5)]
    prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[Tm]] = [xT[1:5];Inf*ones(5)]

    prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[T]] = [x1_sample[i][1:5];-Inf*ones(5)]
    prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[T]] = [x1_sample[i][1:5];Inf*ones(5)]
end

# # add "contact" constraint
# for i = 1:N
#     prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[1]] = [x1_sample[i][1]; -Inf]
#     prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[1]] = [x1_sample[i][1]; Inf]
#
#     prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[models[i].Tm]] .= [0.0; 0.0]
#     prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[models[i].Tm]] .= [0.0; 0.0]
#
#     prob_sample_moi.primal_bounds[1][prob_sample.idx_sample[i].x[T]] .= [x1_sample[i][1]; -Inf]
#     prob_sample_moi.primal_bounds[2][prob_sample.idx_sample[i].x[T]] .= [x1_sample[i][1]; Inf]
# end

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100)
# Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,max_iter=100)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)

Q_nom_sample = [X_nom_sample[t][1:5] for t = 1:T]

foot_traj_nom_sample = [kinematics(model,Q_nom_sample[t]) for t = 1:T]

foot_x_ns = [foot_traj_nom_sample[t][1] for t=1:T]
foot_y_ns = [foot_traj_nom_sample[t][2] for t=1:T]

plt_ft_nom_sample = plot(aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="")

for i = 1:N
    Q_sample = [X_sample[i][t][1:5] for t = 1:T]

    foot_traj_sample = [kinematics(model,Q_sample[t]) for t = 1:T]

    foot_x_s = [foot_traj_sample[t][1] for t=1:T]
    foot_y_s = [foot_traj_sample[t][2] for t=1:T]

    plt_ft_nom_sample = plot!(foot_x_s,foot_y_s,aspect_ratio=:equal,label="")
end

plt_ft_nom_sample = plot!(foot_x_ns,foot_y_ns,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",color=:red,label="nominal")
display(plt_ft_nom_sample)
#
#
# # Simulate controller
# K_sample, X_nom_sample, U_nom_sample, H_nom_sample = controller(Z_sample_sol,prob_sample)
# norm(vec(hcat(K_sample...)) -vec(hcat(K...)))
#
# using Distributions
# model_sim = model
# T_sim = 100*T
#
# μ = zeros(model.nx)
# Σ = Diagonal(1.0e-32*ones(model.nx))
# W = Distributions.MvNormal(μ,Σ)
# w = rand(W,T_sim)
#
# μ0 = zeros(nx)
# Σ0 = Diagonal(1.0e-32*ones(nx))
# W0 = Distributions.MvNormal(μ0,Σ0)
# w0 = rand(W0,1)
#
# z0_sim = vec(copy(X_nominal[1]) + w0)
#
# t_nom = range(0,stop=h0*T,length=T)
# t_sim = range(0,stop=h0*T,length=T_sim)
#
# z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K,X_nominal,U_nominal,
#     model_sim,Q_lqr,R_lqr,T_sim,h0,z0_sim,w,_norm=2,ul=ul,uu=uu)
# z_sample, u_sample, J_sample, Jx_sample, Ju_sample = simulate_linear_controller(K_sample,X_nom_sample,U_nom_sample,
#     model_sim,Q_lqr,R_lqr,T_sim,h0,z0_sim,w,_norm=2,ul=ul,uu=uu)
#
# plt = plot(t_sim,hcat(z_tvlqr...)[1:5,:]',color=:purple,width=2.0,label="")
# plt = plot!(t_sim,hcat(z_sample...)[1:5,:]',color=:orange,width=2.0,label="")
#
# plt = plot(t_sim[1:end-1],hcat(u_tvlqr...)[1:4,:]',color=:purple,label="",width=2.0)
# plt = plot!(t_sim[1:end-1],hcat(u_sample...)[1:4,:]',color=:orange,label="",width=2.0)
#
# Q_sim = [z_tvlqr[t][1:5] for t = 1:T_sim]
#
# foot_traj = [kinematics(model,Q_sim[t]) for t = 1:T_sim]
#
# foot_x = [foot_traj[t][1] for t=1:T_sim]
# foot_y = [foot_traj[t][2] for t=1:T_sim]
#
# plt_ft_nom = plot!(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
#     title="Foot 1 trajectory",label="",color=:purple)
#
# Q_sim_sample = [z_sample[t][1:5] for t = 1:T_sim]
#
# foot_traj_sample = [kinematics(model,Q_sim_sample[t]) for t = 1:T_sim]
#
# foot_x_sample = [foot_traj_sample[t][1] for t=1:T_sim]
# foot_y_sample = [foot_traj_sample[t][2] for t=1:T_sim]
#
# plt_ft_nom_sample = plot!(foot_x_sample,foot_y_sample,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
#     title="Foot 1 trajectory",label="",color=:orange)
#
# # visualize
using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes:Vec,HyperRectangle,HyperSphere,Point3f0,Cylinder
using LinearAlgebra
using MeshCat, MeshCatMechanisms
using MeshIO
using Rotations
using RigidBodyDynamics

urdf = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromleftfoot.urdf"
mechanism = parse_urdf(urdf,floating=false)

vis = Visualizer()
open(vis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf,package_path=[dirname(dirname(urdf))]), vis)

# Q_left = [transformation_to_urdf_left_pinned(z_tvlqr[t][1:5],z_tvlqr[t][6:10])[1] for t = 1:T_sim]
# animation = MeshCat.Animation(mvis,t_sim,Q_left)
# setanimation!(mvis,animation)
for i = 1:T
    set_configuration!(mvis,transformation_to_urdf_left_pinned(X_nominal[i][1:5],X_nominal[i][6:10]))
    sleep(0.1)
end

Q_left = [transformation_to_urdf_left_pinned(X_nominal[t][1:5],X_nominal[t][6:10]) for t = 1:T]
animation = MeshCat.Animation(mvis,range(0,stop=h0*T,length=T),Q_left)
setanimation!(mvis,animation)
