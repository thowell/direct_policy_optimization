include("../src/direct_policy_optimization.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 21
Tm = -1
model.Tm = Tm

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

ϵ = 1.0e-8
θ = 10pi/180
h = model.l2 + model.l1*cos(θ)
ψ = acos(h/(model.l1 + model.l2))
stride = sin(θ)*model.l1 + sin(ψ)*(model.l1+model.l2)
x1 = [π-θ,π+ψ,θ,0,0,0,0,0,0,0]
xT = [π+ψ,π-θ-ϵ,0,θ,0,0,0,0,0,0]
kinematics(model,x1)[1]
kinematics(model,xT)[1]

kinematics(model,x1)[2]
kinematics(model,xT)[2]

q1 = transformation_to_urdf_left_pinned(x1[1:5],x1[1:5])
set_configuration!(mvis,q1)

qT = transformation_to_urdf_left_pinned(xT[1:5],xT[1:5])
set_configuration!(mvis,qT)

ζ = 11
xM = [π,π-ζ*pi/180,0,2*ζ*pi/180,0,0,0,0,0,0]
qM = transformation_to_urdf_left_pinned(xM[1:5],xM[1:5])
set_configuration!(mvis,qM)
kinematics(model,xM)[1]
kinematics(model,xM)[2]

x1_foot_des = kinematics(model,x1)[1]
xT_foot_des = kinematics(model,xT)[1]
xc = 0.5*(x1_foot_des + xT_foot_des)

# r1 = x1_foot_des - xc
r1 = xT_foot_des - xc
r2 = 0.1

zM_foot_des = r2

function z_foot_traj(x)
    sqrt((1.0 - ((x - xc)^2)/(r1^2))*(r2^2))
end

foot_x_ref = range(x1_foot_des,stop=xT_foot_des,length=T)
foot_z_ref = z_foot_traj.(foot_x_ref)

plot(foot_x_ref,foot_z_ref)

@assert norm(Δ(xT)[1:5] - x1[1:5]) < 1.0e-5

# Bounds

# xl <= x <= xu
xl_traj = [-Inf*ones(model.nx) for t = 1:T]
xu_traj = [Inf*ones(model.nx) for t = 1:T]

xl_traj[1][1:5] = x1[1:5]
xu_traj[1][1:5] = x1[1:5]

xl_traj[T][1:5] = xT[1:5]
xu_traj[T][1:5] = xT[1:5]

# ul <= u <= uu
uu = 20.0
ul = -20.0

tf0 = 0.5
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

function discrete_dynamics(model::Biped,x,u,h,t)
    if t == model.Tm
        return rk3(model,Δ(x),u,h)
    else
        return rk3(model,x,u,h)
    end
end

function c_stage!(c,x,u,t,model)
    c[1] = kinematics(model,x[1:5])[2] + 1.0e-4
    nothing
end
#
# function c_stage!(c,x,t,model)
#     c[1] = kinematics(model,x[1:5])[2]
#     nothing
# end

m_stage = 1
stage_ineq = (1:m_stage)

include("../src/loop_delta.jl")

# Objective
qq = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
Q = [t < T ? Diagonal(qq) : Diagonal(qq) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
# penalty_obj = PenaltyObjective(1000.0,foot_z_ref,[t for t = 1:T])
# multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    general_constraints=true,
                    m_general=model.nx,
                    general_ineq=(1:0),
                    stage_constraints=true,
                    m_stage=[t==1 ? 0 : m_stage for t = 1:T-1],
                    stage_ineq=[t==1 ? (1:0) : stage_ineq for t = 1:T-1])

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.001*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal_step = solve(prob_moi,copy(Z0),nlp=:SNOPT,max_iter=100,time_limit=120)

# Unpack solutions
X_nominal_step, U_nominal_step, H_nominal_step = unpack(Z_nominal_step,prob)

Q_nominal_step = [X_nominal_step[t][1:5] for t = 1:T]
plot(hcat(Q_nominal_step...)')
foot_traj = [kinematics(model,Q_nominal_step[t]) for t = 1:T]

foot_x = [foot_traj[t][1] for t=1:T]
foot_z = [foot_traj[t][2] for t=1:T]

plt_ft_nom = plot(foot_x,foot_z,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="",color=:red)

plot(hcat(U_nominal_step...)',linetype=:steppost)

plot(foot_x)
plot(foot_z)

sum(H_nominal_step)

# TVLQR policy
Q_lqr = [t < T ? Diagonal(1.0*ones(model.nx)) : Diagonal(10.0*ones(model.nx)) for t = 1:T]
R_lqr = [Diagonal(1.0e-1*ones(model.nu)) for t = 1:T-1]
H_lqr = [1.0 for t = 1:T-1]

K_lqr = TVLQR_gains(model,X_nominal_step,U_nominal_step,H_nominal_step,Q_lqr,R_lqr)

# Samples
n_policy = model.nu
n_features = model.nx
K0 = [rand(n_policy*n_features) for t = 1:T-1]

N = 2*model.nx
models = [model for i = 1:N]
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
#
# models = [model1,model2,model3,model4,model5,model6,
#           model7,model8,model9,model10,model11,model12,
#           model13,model14,model15,model16,model17,
#           model18,model19,model20]

β = 2.0
w = 0.0*ones(model.nx)
γ = 1.0
# x1_sample = [X_nominal_step[1] for i = 1:N]
x1_sample = resample([X_nominal_step[1] for i = 1:N],β=1.0e-1,w=w)
# xT_gait_sample = resample([xT for i = 1:N],β=β,w=w)
xl_traj_sample = [[-Inf*ones(model.nx) for t = 1:T] for i = 1:N]
xu_traj_sample = [[Inf*ones(model.nx) for t = 1:T] for i = 1:N]

vec(X_nominal_step[1] - x1)

# add "contact" constraint
for i = 1:N
    xl_traj_sample[i][1][1:5] = x1_sample[i][1:5]
    xu_traj_sample[i][1][1:5] = x1_sample[i][1:5]

    # xl_traj_sample[i][models[i].Tm][1:5] = xT_gait_sample[i][1:5]
    # xu_traj_sample[i][models[i].Tm][1:5] = xT_gait_sample[i][1:5]
    #
    # xl_traj_sample[i][T][1:5] = x1_gait_sample[i][1:5]
    # xu_traj_sample[i][T][1:5] = x1_gait_sample[i][1:5]
end

# include("../src/general_constraints.jl")
prob_sample = init_sample_problem(prob,models,Q_lqr,R_lqr,H_lqr,β=β,w=w,γ=γ,
    xl=xl_traj_sample,
    xu=xu_traj_sample,
    n_policy=n_policy,
    n_features=n_features,
    policy_constraint=true,
    disturbance_ctrl=true,
    α=1.0,
    sample_general_constraints=true,
    m_sample_general=N*prob.m_general,
    sample_general_ineq=(1:0)
    )

prob_sample_moi = init_MOI_Problem(prob_sample)

Z0_sample = pack(X_nominal_step,U_nominal_step,H_nominal_step[1],K_lqr,prob_sample)

# Solve
Z_sample_sol = solve(prob_sample_moi,Z0_sample,max_iter=100,nlp=:SNOPT7,time_limit=120)
Z_sample_sol = solve(prob_sample_moi,Z_sample_sol,max_iter=100,nlp=:SNOPT7,time_limit=180)

# Unpack solution
X_nom_sample, U_nom_sample, H_nom_sample, X_sample, U_sample, H_sample = unpack(Z_sample_sol,prob_sample)
K_sample = [Z_sample_sol[prob_sample.idx_K[t]] for t = 1:T-1]

norm(X_nom_sample[1] - Δ(X_nom_sample[T]))
Q_nom_sample = [X_nom_sample[t][1:5] for t = 1:T]

foot_traj_sample = [kinematics(model,Q_nom_sample[t]) for t = 1:T]

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


plt_x = plot()
for i = 1:N
    plt_x = plot!(hcat(X_sample[i]...)[1:5,:]',label="")
end
plt_x = plot!(hcat(X_nom_sample...)[1:5,:]',label="",color=:red)
display(plt_x)

plt_u = plot()
for i = 1:N
    plt_u = plot!(hcat(U_sample[i]...)[1:4,:]',label="",linetype=:steppost)
end
plt_u = plot!(hcat(U_nom_sample...)[1:4,:]',label="",color=:red,linetype=:steppost)
display(plt_u)

# Visualization

q0 = transformation_to_urdf_left_pinned(x1,rand(5))
set_configuration!(mvis,q0)

for i = 1:T
    set_configuration!(mvis,transformation_to_urdf_left_pinned(X_nominal_step[i][1:5],X_nominal_step[i][6:10]))
    sleep(0.1)
end


for i = 1:T
    set_configuration!(mvis,transformation_to_urdf_left_pinned(X_nom_sample[i][1:5],X_nom_sample[i][6:10]))
    sleep(0.1)
end

# Q_left = [transformation_to_urdf_left_pinned(X_nominal[t][1:5],X_nominal[t][6:10]) for t = 1:T]
# animation = MeshCat.Animation(mvis,range(0,stop=h0*T,length=T),Q_left)
# setanimation!(mvis,animation)

using Distributions
model_sim = Biped(0.2755,0.288,Tm,nx,nu)
switch = -1
T_sim = 10*T+1
# model_sim.Tm = convert(Int,(T_sim-1)/2 + 1) + switch

μ = zeros(nx)
Σ = Diagonal(10.0*ones(nx))
W = Distributions.MvNormal(μ,Σ)
w = rand(W,T_sim)

μ0 = zeros(nx)
Σ0 = Diagonal(1.0e-3*ones(nx))
W0 = Distributions.MvNormal(μ0,Σ0)
w0 = rand(W0,1)

t_nominal = range(0,stop=H_nominal_step[1]*(T-1),length=T)
t_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T)

t_sim_nominal = range(0,stop=H_nominal_step[1]*(T-1),length=T_sim)
t_sim_sample = range(0,stop=H_nom_sample[1]*(T-1),length=T_sim)

plt = plot(t_nominal,hcat(X_nominal_step...)[1:5,:]',legend=:bottom,color=:red,label="",
    width=2.0,xlabel="time (s)",title="Biped",ylabel="state")

z_tvlqr, u_tvlqr, J_tvlqr, Jx_tvlqr, Ju_tvlqr = simulate_linear_controller(K_lqr,
    X_nominal_step,U_nominal_step,model_sim,Q_lqr,R_lqr,T_sim,H_nominal_step[1],vec(X_nominal_step[1]+w0),w,_norm=2,
    ul=ul,uu=uu)
plt = plot!(t_sim_nominal,hcat(z_tvlqr...)[1:5,:]',color=:purple,label=["tvlqr" "" "" "" ""],width=2.0)

plt = plot(t_sample,hcat(X_nom_sample...)[1:5,:]',legend=:bottom,color=:red,label="",
    width=2.0,xlabel="time (s)",title="Biped",ylabel="state")
z_sample, u_sample, J_sample,Jx_sample, Ju_sample = simulate_linear_controller(K_sample,
    X_nom_sample,U_nom_sample,model_sim,Q_lqr,R_lqr,T_sim,H_nom_sample[1],vec(X_nom_sample[1]+w0),w,_norm=2,
    ul=ul,uu=uu,controller=:policy)
plt = plot!(t_sim_sample,hcat(z_sample...)[1:5,:]',linetype=:steppost,color=:orange,label=["sample" "" "" "" ""],width=2.0)

# objective value
J_tvlqr
J_sample

# state tracking
Jx_tvlqr
Jx_sample

# control tracking
Ju_tvlqr
Ju_sample
