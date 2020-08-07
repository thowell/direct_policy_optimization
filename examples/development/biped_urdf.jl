include("../dynamics/biped.jl")
include("../dynamics/biped/utils.jl")

using RigidBodyDynamics
using MeshCat, MeshCatMechanisms
urdf_left = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromleftfoot.urdf"
mechanism_left = parse_urdf(urdf_left,floating=false)

urdf_right = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromrightfoot.urdf"
mechanism_right = parse_urdf(urdf_right,floating=false)

vis = Visualizer()
open(vis)
mvis_left = MechanismVisualizer(mechanism_left, URDFVisuals(urdf_left,package_path=[dirname(dirname(urdf_left))]), vis)
mvis_right = MechanismVisualizer(mechanism_right, URDFVisuals(urdf_right,package_path=[dirname(dirname(urdf_right))]), vis)

# animation = MeshCat.Animation(mvis_right,t_nominal[Tm+1:T],Q_nominal_urdf_right)

animation = MeshCat.Animation(mvis_left,t_nominal,Q_nominal_urdf_left)
setanimation!(mvis_left,animation)

# Initial and final states
q_init = [2.44917526290273,2.78819438807838,0.812693850088907,0.951793806080012,1.49974183648719e-13]
v_init = [0.0328917694318260,0.656277832705193,0.0441573173000750,1.03766701983449,1.39626340159558]
q_final = [2.84353387648829,2.48652580597252,0.751072212241267,0.645830978766432, 0.0612754113212848]
v_final = [2.47750069399969,3.99102008940145,-1.91724136219709,-3.95094757056324,0.0492401787458546]

x1 = [q_init;v_init]
xT = Δ(x1)#[q_final;v_final]

q1_left = transformation_to_urdf_left_pinned(x1[1:5],x1[6:10])
qT_left = transformation_to_urdf_left_pinned(xT[1:5],xT[6:10])
q1_right = transformation_to_urdf_right_pinned(q_init,v_init)
qT_right = transformation_to_urdf_right_pinned(q_final,v_final)

set_configuration!(mvis_left,qT_left)
set_configuration!(mvis_right,q1_right)

# # left
# state_left = MechanismState(mechanism_left)
#
# body1 = findbody(mechanism_left, "left_shank")
# point1 = Point3D(default_frame(body1), 0.0, 0.0, 0.0)
# setelement!(mvis_left, point1, 0.025)
#
# body2 = findbody(mechanism_left, "right_shank")
# point2 = Point3D(default_frame(body2), 0.288, 0.0, 0.0)
# setelement!(mvis_left, point2, 0.05)
#
# world = root_frame(mechanism_left)
# set_configuration!(state_left,q1_left)
#
# point1_in_world = transform(state_left, point1, world).v
# point2_in_world = transform(state_left, point2, world).v
#
# # right
# state_right = MechanismState(mechanism_right)
#
# body1r = findbody(mechanism_right, "left_shank")
# point1r = Point3D(default_frame(body1r), 0.288, 0.0, 0.0)
# setelement!(mvis_right, point1r, 0.05)
#
# body2r = findbody(mechanism_right, "right_shank")
# point2r = Point3D(default_frame(body2r), 0.0, 0.0, 0.0)
# setelement!(mvis_right, point2r, 0.025)
#
# worldr = root_frame(mechanism_right)
# set_configuration!(state_right,q1_right)
#
# point1_in_world = transform(state_right, point1r, worldr).v
# point2_in_world = transform(state_right, point2r, worldr).v
