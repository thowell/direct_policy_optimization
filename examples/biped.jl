include("../src/sample_trajectory_optimization.jl")
include("../dynamics/biped.jl")
using Plots

# Horizon
T = 20

# Bounds

# ul <= u <= uu
uu = 20.0
ul = -20.0

tf0 = .36915
h0 = tf0/(T-1)
hu = h0
hl = h0

# Initial and final states
q_init = [2.44917526290273,2.78819438807838,0.812693850088907,0.951793806080012,1.49974183648719e-13]
v_init = [0.0328917694318260,0.656277832705193,0.0441573173000750,1.03766701983449,1.39626340159558]
q_final = [2.84353387648829,2.48652580597252,0.751072212241267,0.645830978766432, 0.0612754113212848]
v_final = [2.47750069399969,3.99102008940145,-1.91724136219709,-3.95094757056324,0.0492401787458546]

x1 = [q_init;v_init]
xT = [q_final;v_final]

pfz_init = kinematics(model,q_init)[2]

function c_stage!(c,x,u,t,model)
    c[1] = kinematics(model,x[1:5])[2] - pfz_init
    nothing
end

m_stage = 1

# Objective
Q = [t < T ? Diagonal([zeros(5);zeros(5)]) : Diagonal(zeros(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-3*ones(model.nu)) for t = 1:T-1]
c = 0.0
obj = QuadraticTrackingObjective(Q,R,c,
    [xT for t=1:T],[zeros(model.nu) for t=1:T-1])
penalty_obj = PenaltyObjective(1.0,0.05)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,x1,xT,model,multi_obj,
                    ul=[ul*ones(model.nu) for t=1:T-1],
                    uu=[uu*ones(model.nu) for t=1:T-1],
                    hl=[hl for t=1:T-1],
                    hu=[hu for t=1:T-1],
                    goal_constraint=true,
                    stage_constraints=true,
                    m_stage=[m_stage for t=1:T-1]
                    )

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
foot_traj = [kinematics(model,Q_nominal[t]) for t = 1:T]

# Time trajectories
t_nominal = zeros(T)
for t = 2:T
    t_nominal[t] = t_nominal[t-1] + H_nominal[t-1]
end

# Visualize trajectory
# using RigidBodyDynamics
# using MeshCat, MeshCatMechanisms
# urdf = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromleftfoot.urdf"
# mechanism = parse_urdf(urdf,floating=false)
# vis = Visualizer()
# open(vis)
# mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf,package_path=[dirname(dirname(urdf))]), vis)

animation = MeshCat.Animation(mvis,t_nominal,Q_nominal_urdf_left)
setanimation!(mvis,animation)

# control trajectory
plot(t_nominal[1:end-1],hcat(U_nominal...)',width=2.0,linetype=:steppost,
    xlabel="time (s)",title="control",label=["knee 1" "hip 1" "hip 2" "knee 1"],
    legend=:bottom)

foot_x = [foot_traj[t][1] for t=1:T]
foot_y = [foot_traj[t][2] for t=1:T]

plot(foot_x,foot_y,aspect_ratio=:equal,xlabel="x",ylabel="z",width=2.0)
