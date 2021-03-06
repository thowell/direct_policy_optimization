include("../src/direct_policy_optimization.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 21
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
x_ref = [π;π - π/4.0;0.0;π/2.0;0.0;0.0;0.0;0.0;0.0;0.0]

r1 = 0.05
r2 = 0.01
x1_des = -r1
yM_des = r2

function y_traj(x)
    sqrt((1.0 - (x^2)/(r1^2))*(r2^2))
end

x = range(-r1,stop=r1,length=T)
y = y_traj.(x)

plot(x,y)

function general_constraints!(c,Z,prob::TrajectoryOptimizationProblem)
	c[1] = kinematics(prob.model,Z[prob.idx.x[1]])[1] + r1
	c[2] = kinematics(prob.model,Z[prob.idx.x[1]])[2]
	c[3] = kinematics(prob.model,Z[prob.idx.x[prob.T]])[1] - r1
	c[4] = kinematics(prob.model,Z[prob.idx.x[prob.T]])[2]
	nothing
end

function ∇general_constraints!(∇c,Z,prob::TrajectoryOptimizationProblem)
	tmp1(y) = [kinematics(prob.model,y)[1] + r1]
	tmp2(y) = [kinematics(prob.model,y)[2]]
	tmp3(y) = [kinematics(prob.model,y)[1] - r1]
	tmp4(y) = [kinematics(prob.model,y)[2]]

	s = 0
	r_idx = 1:1
	c_idx = prob.idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[1:len] = ForwardDiff.jacobian(tmp1,Z[prob.idx.x[1]])
	s += len

	r_idx = 1 .+ (1:1)
	c_idx = prob.idx.x[1]
	len = length(r_idx)*length(c_idx)
	∇c[1:len] = ForwardDiff.jacobian(tmp2,Z[prob.idx.x[1]])
	s += len

	r_idx = 2 .+ (1:1)
	c_idx = prob.idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[s .+ (1:len)] = ForwardDiff.jacobian(tmp3,Z[prob.idx.x[T]])
	s += len

	r_idx = 3 .+ (1:1)
	c_idx = prob.idx.x[T]
	len = length(r_idx)*length(c_idx)
	∇c[s .+ (1:len)] = ForwardDiff.jacobian(tmp4,Z[prob.idx.x[T]])
	s += len
	nothing
end

function general_constraint_sparsity(prob::TrajectoryOptimizationProblem;
		r_shift=0,)
	row = []
	col = []

	s = 0
	r_idx = r_shift .+ (1:1)
	c_idx = prob.idx.x[1]
	row_col!(row,col,r_idx,c_idx)

	r_idx = r_shift + 1 .+ (1:1)
	c_idx = prob.idx.x[1]
	row_col!(row,col,r_idx,c_idx)

	r_idx = r_shift + 2 .+ (1:1)
	c_idx = prob.idx.x[T]
	row_col!(row,col,r_idx,c_idx)

	r_idx = r_shift + 3 .+ (1:1)
	c_idx = prob.idx.x[T]
	row_col!(row,col,r_idx,c_idx)

	return collect(zip(row,col))
end

# Bounds

# xl <= x <= xu
xl_traj = [-Inf*ones(model.nx) for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

# xl_traj[1] = x1
# xu_traj[1] = x1
#
# xl_traj[T] = xT
# xu_traj[T] = xT

# ul <= u <= uu
uu = 100*20.0
ul = -100*20.0

tf0 = 0.36915
h0 = tf0/(T-1)
hu = 5.0*h0
hl = 0.0*h0

function discrete_dynamics(model::Biped,x⁺,x,u,h,t)
    if t == model.Tm
        return rk3_implicit(model,x⁺,Δ(x),u,h)
    else
        return rk3_implicit(model,x⁺,x,u,h)
    end
end

function c_stage!(c,x,u,t,model)
    if t != 1
        c[1] = kinematics(model,x[1:5])[2]
    else
        c[1] = 0.0
    end
    nothing
end

function c_stage!(c,x,t,model)
    c[1] = kinematics(model,x[1:5])[2]
    nothing
end

m_stage = 1
stage_ineq = (1:m_stage)

# Objective
qq = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
R = [Diagonal(1.0e-2*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [x_ref for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(100.0,y,[t for t = 1:T])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
					general_constraints=true,
					m_general=4,
					general_ineq=(1:0),
                    stage_constraints=true,
                    m_stage=[m_stage for t = 1:T-1],
                    stage_ineq=[stage_ineq for t = 1:T-1])

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:ipopt,max_iter=500,time_limit=30)

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

Q_nominal = [X_nominal[t][1:5] for t = 1:T]
plot(hcat(Q_nominal...)')
foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=1:T]
foot_y = [foot_traj[t][2] for t=1:T]

plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)

# # start in mid trajectory
# Tm = convert(Int,(T-1)/2 + 1)
# model.Tm = Tm
# x1 = X_nominal[Tm+3]
#
# tf0 = 0.5*0.36915
# h0 = tf0/(T-1)
# hu = 5.0*h0
# hl = 0.0*h0
#
# xl_traj = [-Inf*ones(model.nx) for t = 1:T]
# xu_traj = [Inf*ones(model.nx) for t = 1:T]
#
# xl_traj[1][1:5] = x1[1:5]
# xu_traj[1][1:5] = x1[1:5]
#
# xl_traj[Tm][1:5] = xT[1:5]
# xu_traj[Tm][1:5] = xT[1:5]
#
# xl_traj[T][1:5] = x1[1:5]
# xu_traj[T][1:5] = x1[1:5]
#
# Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
# R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
# c = 1.0
# obj = QuadraticTrackingObjective(Q,R,c,
#     [x1 for t=1:T],[zeros(model.nu) for t=1:T-1])
# penalty_obj = PenaltyObjective(1.0e-1,0.1,[t for t = 1:T-1])
# multi_obj = MultiObjective([obj,penalty_obj])
#
# # Problem
# include("../src/loop.jl")
# prob = init_problem(model.nx,model.nu,T,model,multi_obj,
#                     xl=xl_traj,
#                     xu=xu_traj,
#                     ul=[ul*ones(model.nu) for t=1:T-1],
#                     uu=[uu*ones(model.nu) for t=1:T-1],
#                     hl=[hl for t=1:T-1],
#                     hu=[hu for t=1:T-1],
#                     stage_constraints=true,
#                     m_stage=[m_stage for t = 1:T-1],
#                     stage_ineq=[stage_ineq for t = 1:T-1],
#                     general_constraints=true,
#                     m_general=model.nx,
#                     general_ineq=(1:0))
#
# # MathOptInterface problem
# prob_moi = init_MOI_Problem(prob)
#
# # Trajectory initialization
# X0 = [linear_interp(x1,xT,Tm)...,linear_interp(x1,xT,Tm-1)...] # linear interpolation on state
# # X0 = linear_interp(x1,xT,T) # linear interpolation on state
#
# U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls
#
# # Pack trajectories into vector
# Z0 = pack(X0,U0,h0,prob)
#
# # Solve nominal problem
# @time Z_nominal = solve(prob_moi,copy(Z0),nlp=:SNOPT7)
#
# # Unpack solutions
# X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)
#
# @assert norm(X_nominal[1][1:5] - X_nominal[T][1:5]) < 1.0e-5
# @assert norm(X_nominal[Tm][1:5] - xT[1:5]) < 1.0e-5
#
# Q_nominal = [X_nominal[t][1:5] for t = 1:T]
#
# foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]
#
# foot_x = [foot_traj[t][1] for t=(1:Tm)]
# foot_y = [foot_traj[t][2] for t=(1:Tm)]
# plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
#     title="Foot 1 trajectory",label="",color=:red)
# @show foot_x[1]
# @show foot_x[end]
#
# foot_x = [kinematics(model,Δ(X_nominal[Tm]))[1],[foot_traj[t][1] for t=(Tm+1):T]...]
# foot_y = [kinematics(model,Δ(X_nominal[Tm]))[2],[foot_traj[t][2] for t=(Tm+1):T]...]
# @show foot_x[end]
#
# kinematics(model,xT[1:5])
# plt_ft_nom = plot!(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
#     title="Foot 1 trajectory",label="",color=:blue)
#
# # foot_x = [foot_traj[t][1] for t=(1:T)]
# # foot_y = [foot_traj[t][2] for t=(1:T)]
# # plt_ft_nom = plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
# #     title="Foot 1 trajectory",label="",color=:red)

# TVLQR policy
Q_lqr = [t < T ? Diagonal(1.0*ones(model.nx)) : Diagonal(1.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(0.1*ones(model.nu)) for t = 1:T-1]
H_lqr = [1.0 for t = 1:T-1]

K = TVLQR_gains(model,X_nominal,U_nominal,H_nominal,Q_lqr,R_lqr)

# Samples
N = 2*model.nx
models=[model for i = 1:N]
# model1 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model2 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model3 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model4 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model5 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model6 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model7 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model8 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model9 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model10 = Biped(0.2755,0.288,Tm-1,nx,nu)
# model11 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model12 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model13 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model14 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model15 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model16 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model17 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model18 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model19 = Biped(0.2755,0.288,Tm+1,nx,nu)
# model20 = Biped(0.2755,0.288,Tm+1,nx,nu)

# models = [model1,model2,model3,model4,model5,model6,
#           model7,model8,model9,model10,model11,model12,
#           model13,model14,model15,model16,model17,
#           model18,model19,model20]

# K0 = [rand(model.nu,model.nx) for t = 1:T-1]
β = 1.0
w = 1.0e-5*ones(model.nx)
γ = 1.0
x1_sample = resample([x1 for i = 1:N],β=β,w=w)

xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

# add "contact" constraint
for i = 1:N
    xl_traj_sample[i][1] = x1_sample[i]
    xu_traj_sample[i][1] = x1_sample[i]

    # xl_traj_sample[i][models[i].Tm] = xT
    # xu_traj_sample[i][models[i].Tm] = xT
    #
    xl_traj_sample[i][T] = xT
    xu_traj_sample[i][T] = xT
end

prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    policy_constraint=true,
    disturbance_ctrl=true,
    α=1.0
    )


prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal,U_nominal,H_nominal[1],K,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100,nlp=:SNOPT7,time_limit=180)
Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,max_iter=100,nlp=:SNOPT7,time_limit=180)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample = unpack(Z_sample_sol,prob_sample)

Q_nom_sample = [X_nom_sample[t][1:5] for t = 1:T]

foot_traj_sample = [kinematics(model,Q_nom_sample[t]) for t = 1:T]
@show foot_x_sample[1]
@show foot_x_sample[end]

plt_ft_nom = plot()
for i = 1:N
    Q_nom_sample = [X_sample[i][t][1:5] for t = 1:T]

    foot_traj_sample = [kinematics(model,Q_nom_sample[t]) for t = 1:T]

    foot_x_sample = [foot_traj_sample[t][1] for t=(1:T)]
    foot_y_sample = [foot_traj_sample[t][2] for t=(1:T)]
    plt_ft_nom = plot!(foot_x_sample,foot_y_sample,aspect_ratio=:equal,xlabel="x",ylabel="z",
        title="Foot 1 trajectory",label="")
    @show foot_x_sample[1]
    @show foot_x_sample[end]
end
foot_x_sample = [foot_traj_sample[t][1] for t=(1:T)]
foot_y_sample = [foot_traj_sample[t][2] for t=(1:T)]
plt_ft_nom = plot!(foot_x_sample,foot_y_sample,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)

display(plt_ft_nom)

# Visualization
using Colors
using CoordinateTransformations
using FileIO
using GeometryTypes:Vec,HyperRectangle,HyperSphere,Point3f0,Cylinder
using LinearAlgebra
using MeshCat, MeshCatMechanisms
using MeshIO
using Rotations
using RigidBodyDynamics

urdf = "/home/taylor/Research/direct_policy_optimization/dynamics/biped/urdf/flip_5link_fromleftfoot.urdf"
mechanism = parse_urdf(urdf,floating=false)

vis = Visualizer()
open(vis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf,package_path=[dirname(dirname(urdf))]), vis)

q0 = transformation_to_urdf_left_pinned(x1,rand(5))
set_configuration!(mvis,q0)

for i = 1:T
    set_configuration!(mvis,transformation_to_urdf_left_pinned(X_nom_sample[i][1:5],X_nom_sample[i][6:10]))
    sleep(0.2)
end

# Q_left = [transformation_to_urdf_left_pinned(X_nominal[t][1:5],X_nominal[t][6:10]) for t = 1:T]
# animation = MeshCat.Animation(mvis,range(0,stop=h0*T,length=T),Q_left)
# setanimation!(mvis,animation)
