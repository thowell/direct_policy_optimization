include("biped/utils.jl")
using RigidBodyDynamics, SparseArrays
urdf = "/home/taylor/Research/sample_trajectory_optimization/dynamics/biped/urdf/flip_5link_fromleftfoot.urdf"
mechanism = parse_urdf(urdf,floating=false)

mutable struct BipedRBD
    state
	result
	B
	point1
	point2
	world
    nx::Int
    nu::Int
end

function dynamics_rbd_x(model,x,u)
	idx_q = model.idx_q
	idx_v = model.idx_v

	T = eltype(x)

	state = model.state[T]
	result = model.result[T]

	set_configuration!(state,x[idx_q])
	set_velocity!(state,x[idx_v])

	dynamics!(result, state, model.B*u)

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

	dynamics!(result, state, model.B*u)

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

	dynamics!(result, state, model.B*u)

	return [result.q̇; result.v̇]
end

function dynamics_rbd(model,x,u)
	if eltype(x) <: ForwardDiff.Dual
		return dynamics_rbd_x(model,x,u)
	elseif eltype(u) <: ForwardDiff.Dual
		return dynamics_rbd_u(model,x,u)
	else
		return dynamics_rbd_(model,x,u)
	end
end

state = MechanismState(mechanism)
result = DynamicsResult(mechanism)

nx = 10
nu = 4

B = spzeros(nx,nu)
B[2,1] = 1.0
B[3,2] = 1.0
B[4,3] = 1.0
B[5,4] = 1.0

body1 = findbody(mechanism, "left_shank")
point1 = Point3D(default_frame(body1), 0.0, 0.0, 0.0)

body2 = findbody(mechanism, "right_shank")
point2 = Point3D(default_frame(body2), 0.288, 0.0, 0.0)

world = root_frame(mechanism)

model = BipedRBD(state,result,B,point1,point2,world,nx,nu)

function kinematics(model::BipedRBD,q)
	set_configuration!(model.state,q)

	point2_in_world = transform(state,point2,world).v

	return [point2_in_world[1],point2_in_world[3]]
end

mutable struct PenaltyObjective{T} <: Objective
    α::T
    pfz_des::T
end

function objective(Z,l::PenaltyObjective,model::BipedRBD,idx,T)
    J = 0
    for t = 1:T-1
        q = Z[idx.x[t][1:5]]
        pfz = kinematics(model,q)[2]
        J += (pfz - l.pfz_des)*(pfz - l.pfz_des)
    end
    return l.α*J
end

function objective_gradient!(∇l,Z,l::PenaltyObjective,model::BipedRBD,idx,T)
    for t = 1:T-1
        q = Z[idx.x[t][1:5]]
        tmp(w) = kinematics(model,w)[2]
        pfz = tmp(q)
        ∇l[idx.x[t][1:5]] += 2.0*l.α*(pfz - l.pfz_des)*ForwardDiff.gradient(tmp,q)
    end
    return nothing
end
