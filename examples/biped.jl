include("../src/sample_trajectory_optimization.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 10
Tm = 5

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
xl_traj = [t != Tm ? -Inf*ones(model.nx) : xT for t = 1:T]
xu_traj = [t != Tm ? Inf*ones(model.nx) : xT for t = 1:T]

# ul <= u <= uu
uu = 20.0
ul = -20.0

tf0 = 2.0*0.36915
h0 = tf0/(T-1)
hu = h0
hl = h0

function discrete_dynamics(model::Biped,x⁺,x,u,h,t)
    if t == model.Tm
        rk3_implicit(model,x⁺,Δ(x),u,h)
    else
        return rk3_implicit(model,x⁺,x,u,h)
    end
end

# pfz_init = kinematics(model,q_init)[2]
#
# function c_stage!(c,x,u,t,model)
#     c[1] = kinematics(model,x[1:5])[2] - pfz_init
#     nothing
# end
#
# m_stage = 1

# Objective
Q = [t < T ? Diagonal([zeros(5);zeros(5)]) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-3*ones(model.nu)) for t = 1:T-1]
c = 1.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1.0,0.05,[t for t = 1:T-1 if (t != Tm-1 || t != 1)])
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true,
                    stage_constraints=false)#,
                    # m_stage=[t==Tm ? 0 : m_stage for t=1:T-1]
                    # )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = linear_interp(x1,xT,T) # linear interpolation on state
U0 = [0.1*rand(model.nu) for t = 1:T-1] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,h0,prob)

# Solve nominal problem
@time Z_nominal = solve(prob_moi,copy(Z0))

# Unpack solutions
X_nominal, U_nominal, H_nominal = unpack(Z_nominal,prob)

Q_nominal = [X_nominal[t][1:5] for t = 1:T]
Q_nominal_urdf_left = [transformation_to_urdf_left_pinned(X_nominal[t][1:5],X_nominal[t][5 .+ (1:5)])[1] for t = 1:T]
# Q_nominal_urdf_right = [transformation_to_urdf_right_pinned(X_nominal[t][1:5],X_nominal[t][5 .+ (1:5)]) for t = Tm+1:T]

foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]
foot_traj_Tm = kinematics(model,Δ(X_nominal[Tm]))
xz_traj = [pe(model,Q_nominal[t]) for t=1:T]

plot(hcat(xz_traj...)[1:1,1:Tm]')
plot!(hcat(xz_traj...)[2:2,1:Tm]')

traj_2D = [[xz_traj[t][1:2];Q_nominal[t][5];Q_nominal[t][1];Q_nominal[t][3];Q_nominal[t][2];Q_nominal[t][4]] for t = 1:Tm]

# Time trajectories
t_nominal = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
end

# Visualize trajectory
using RigidBodyDynamics
using MeshCat, MeshCatMechanisms
urdf_left = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromleftfoot.urdf"
mechanism_left = parse_urdf(urdf_left,floating=false)
#
# urdf_right = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromrightfoot.urdf"
# mechanism_right = parse_urdf(urdf_right,floating=false)

vis = Visualizer()
open(vis)
mvis_left = MechanismVisualizer(mechanism_left, URDFVisuals(urdf_left,package_path=[dirname(dirname(urdf_left))]), vis)
# mvis_right = MechanismVisualizer(mechanism_right, URDFVisuals(urdf_right,package_path=[dirname(dirname(urdf_right))]), vis)

set_configuration!(mvis_left,Q_nominal_urdf_left[1])
# set_configuration!(mvis_right,Q_nominal_urdf_right[1])

# animation = MeshCat.Animation(mvis_right,t_nominal[Tm+1:T],Q_nominal_urdf_right)

animation = MeshCat.Animation(mvis_left,t_nominal,Q_nominal_urdf_left)
setanimation!(mvis_left,animation)

# control trajectory
plot(t_nominal[1:end-1],hcat(U_nominal...)',width=2.0,linetype=:steppost,
    xlabel="time (s)",title="Control",label=["knee 1" "hip 1" "hip 2" "knee 1"],
    )

foot_x = [foot_traj[t][1] for t=1:T]
foot_y = [foot_traj[t][2] for t=1:T]

plt1 = plot(foot_x[1:Tm],foot_y[1:Tm],aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 1 trajectory",label="")
plt2 = plot([foot_traj_Tm[1],foot_x[Tm+1:end]...],[foot_traj_Tm[2],foot_y[Tm+1:end]...],aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0,
    title="Foot 2 trajectory",label="")
plot(plt1,plt2,layout=(2,1))

kinematics(model,X_nominal[end])
sum(H_nominal)
kinematics(model,x1[1:5])
kinematics(model,Δ(xT)[1:5])

plot(t_nominal,foot_x,label="x(t)")
plot!(t_nominal,foot_y,label="z(t)")
