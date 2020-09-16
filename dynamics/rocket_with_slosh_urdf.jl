include(joinpath(pwd(),"src/direct_policy_optimization.jl"))

using RigidBodyDynamics
using MeshCat, MeshCatMechanisms, Random, Blink
urdf = joinpath(pwd(),"dynamics/floating_base_double_pendulum.urdf")
mechanism = parse_urdf(urdf,floating=false)

vis = Visualizer()
open(vis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf,package_path=[dirname(dirname(urdf))]), vis)

state = MechanismState(mechanism)
q = copy(RigidBodyDynamics.configuration(state))
q[1] = 1.0
q[2] = 0.5
q[3] = pi/3
# q[4] = pi - pi/6
set_configuration!(state,q)
set_configuration!(mvis,q)

shoulder = joints(mechanism)[1]
elbow = joints(mechanism)[2]
world = root_frame(mechanism)

body1 = findbody(mechanism, "body")
point1 = Point3D(default_frame(body1), 0., 0, -0.5)
point_jacobian_frame1 = point1.frame
point_jacobian_path1 = path(mechanism, root_body(mechanism), body1)
setelement!(mvis, point1, 0.07)

body2 = findbody(mechanism, "fuel")
point2 = Point3D(default_frame(body2), 0., 0, 0.1)
point_jacobian_frame2 = point2.frame
point_jacobian_path2 = path(mechanism, root_body(mechanism), body2)
setelement!(mvis, point2, 0.07)
