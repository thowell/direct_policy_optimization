include(joinpath(pwd(),"src/direct_policy_optimization.jl"))

using RigidBodyDynamics
using MeshCat, MeshCatMechanisms, Random, Blink
urdf = joinpath(pwd(),"dynamics/floating_base_2D.urdf")
mechanism = parse_urdf(urdf,floating=false)

vis = Visualizer()
open(vis)
mvis = MechanismVisualizer(mechanism, URDFVisuals(urdf,package_path=[dirname(dirname(urdf))]), vis)

state = MechanismState(mechanism)
q = copy(RigidBodyDynamics.configuration(state))
q[1] = 1.0
q[2] = 1.0
q[3] = pi/6
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

state = MechanismState(mechanism)
state_cache = StateCache(mechanism)
result = DynamicsResult(mechanism)
result_cache = DynamicsResultCache(mechanism)

struct RocketURDF{T}
	nx
    nu
    mr
    lr

    state
    result
end

nx = 6
nu = 2
mr = 1.0
lr = 1.0

B(model::RocketURDF,q) = [1.0 0.0;
        0.0 1.0;
        0.5*lr*cos(q[3]) -0.5*lr*sin(q[3])]

function dynamics_rbd_x(model,x,u)
	idx_q = model.idx_q
	idx_v = model.idx_v

	T = eltype(x)

	state = model.state[T]
	result = model.result[T]

	set_configuration!(state,x[idx_q])
	set_velocity!(state,x[idx_v])

	dynamics!(result, state, B(model,x[idx_q])*u)

	return [result.q̇; result.v̇]
end

function dynamics_rbd_u(model,x,u)
	idx_q = model.idx_q
	idx_v = model.idx_v

	T = eltype(u)

	state = model.state[T]
	result = model.result[T]

	set_configuration!(state,x[idx_q])
	set_velocity!(state,x[idx_v])

	dynamics!(result, state, B(model,x[idx_q])*u)

	return [result.q̇; result.v̇]
end

function dynamics_rbd_(model,x,u)
	idx_q = model.idx_q
	idx_v = model.idx_v

	T = eltype(x)
	state = model.state[T]
	result = model.result[T]

	set_configuration!(state,x[idx_q])
	set_velocity!(state,x[idx_v])

	dynamics!(result, state, B(model,x[idx_q])*u)

	return [result.q̇; result.v̇]
end

function dynamics(model::RocketURDF,x,u)
	if eltype(x) <: ForwardDiff.Dual
		return dynamics_rbd_x(model,x,u,w)
	elseif eltype(u) <: ForwardDiff.Dual
		return dynamics_rbd_u(model,x,u,w)
	else
		return dynamics_rbd_(model,x,u,w)
	end
end
