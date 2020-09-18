using MeshCat,MeshCatMechanisms,RigidBodyDynamics
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, MeshCat

urdf_original = joinpath(pwd(),"models/kuka/kuka.urdf")
urdf_new = joinpath(pwd(),"models/kuka/temp/kuka.urdf")

function write_kuka_urdf()
    kuka_mesh_dir = joinpath(pwd(),"models/kuka/meshes")
    temp_dir = joinpath(pwd(),"models/kuka/temp")
    if !isdir(temp_dir)
        mkdir(temp_dir)
    end
    open(urdf_original,"r") do f
        open(urdf_new, "w") do fnew
            for ln in eachline(f)
                pre = findfirst("<mesh filename=",ln)
                post = findlast("/>",ln)
                if !(pre isa Nothing) && !(post isa Nothing)
                    inds = pre[end]+2:post[1]-2
                    pathstr = ln[inds]
                    file = splitdir(pathstr)[2]
                    ln = ln[1:pre[end]+1] * joinpath(kuka_mesh_dir,file) * ln[post[1]-1:end]
                end
                println(fnew,ln)
            end
        end
    end
end

write_kuka_urdf()

function jacobian_transpose_ik!(state::MechanismState,
                               body::RigidBody,
                               point::Point3D,
                               desired::Point3D;
                               α=0.1,
                               iterations=100,
                               visualize=false,
                               mvis="")
    mechanism = state.mechanism
    world = root_frame(mechanism)

    # Compute the joint path from world to our target body
    p = path(mechanism, root_body(mechanism), body)
    # Allocate the point jacobian (we'll update this in-place later)
    Jp = point_jacobian(state, p, transform(state, point, world))

    q = copy(configuration(state))

    for i in 1:iterations
        # Update the position of the point
        point_in_world = transform(state, point, world)
        # Update the point's jacobian
        point_jacobian!(Jp, state, p, point_in_world)
        # Compute an update in joint coordinates using the jacobian transpose
        Δq = α * Array(Jp)' * (transform(state, desired, world) - point_in_world).v
        # Apply the update
        q .= configuration(state) .+ Δq
        set_configuration!(state, q)
        visualize && set_configuration!(mvis, q)
    end
    q
end

urdf_path = joinpath(pwd(),"models/kuka/temp/kuka.urdf")

kuka = MeshCatMechanisms.parse_urdf(urdf_path,remove_fixed_tree_joints=true)
kuka_visuals = MeshCatMechanisms.URDFVisuals(urdf_path)

state = MechanismState(kuka)
state_cache = StateCache(kuka)
result = DynamicsResult(kuka)
result_cache = DynamicsResultCache(kuka)

vis = Visualizer()

mvis = MechanismVisualizer(kuka, kuka_visuals, vis[:base])

open(vis)

nq = num_positions(kuka)
q = zeros(nq)
q[1] = pi/4
q[2] = pi/4

q[4] = -pi/2
# q[6] = pi/2
# q[7] = pi/2
set_configuration!(state,q)
set_configuration!(mvis, q)
ee = findbody(kuka, "iiwa_link_7")
ee_point = Point3D(default_frame(ee), 0.0, 0.0, 0.0)
setelement!(mvis, ee_point, 0.055)

world = root_frame(kuka)
ee_in_world = transform(state, ee_point, world).v
desired = Point3D(world,0.25,0.25,0.0)
q_res = jacobian_transpose_ik!(state,ee,ee_point,desired,
    visualize=true,mvis=mvis)
set_configuration!(mvis, q_res)

desired = Point3D(world,0.75,0.75,0.0)
q_res = jacobian_transpose_ik!(state,ee,ee_point,desired,
    visualize=true,mvis=mvis)
set_configuration!(mvis, q_res)

using StaticArrays

# Kuka iiwa arm parsed from URDF using RigidBodyDynamics.jl
mutable struct KukaParticle{T}
    qL::Vector{T}
    qU::Vector{T}

    state_cache1
    state_cache2
    state_cache3

    result_cache1
    result_cache2
    result_cache3

    world

    nx
    nu
end


results_cache1 = DynamicsResultCache(kuka)
results_cache2 = DynamicsResultCache(kuka)
results_cache3 = DynamicsResultCache(kuka)

state_cache1 = StateCache(kuka)
state_cache2 = StateCache(kuka)
state_cache3 = StateCache(kuka)

# Dimensions
nq = 7 # configuration dim
nu = 7 # control dim

# Methods
function M_func(m::Kuka,q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state,q)
    mass_matrix!(result.massmatrix,state)
    return result.massmatrix
end

function dynamics_bias(m::Kuka,qk,qn,h)
    if eltype(qk) <: ForwardDiff.Dual
		return dynamics_bias_qk(m,qk,qn,h)
	elseif eltype(qn) <: ForwardDiff.Dual
		return dynamics_bias_qn(m,qk,qn,h)
	elseif eltype(h) <: ForwardDiff.Dual
		return dynamics_bias_h(m,qk,qn,h)
	else
		return _dynamics_bias(m,qk,qn,h)
	end
    return result.dynamicsbias
end

function _dynamics_bias(m::Kuka,qk,qn,h)
	T = eltype(m.qL)
	state = m.state_cache1[T]
	result = m.result_cache1[T]
	set_configuration!(state,qk)
	set_velocity!(state,(qn-qk)/h[1])
	dynamics_bias!(result,state)
    return result.dynamicsbias
end

function dynamics_bias_qk(m::Kuka,qk::Vector{T},qn,h) where T
    state = m.state_cache2[T]
    result = m.result_cache2[T]
    set_configuration!(state,qk)
    set_velocity!(state,(qn-qk)/h[1])
    dynamics_bias!(result,state)
    return result.dynamicsbias
end

function dynamics_bias_qn(m::Kuka,qk,qn::Vector{T},h) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state,qk)
    set_velocity!(state,(qn-qk)/h[1])
    dynamics_bias!(result,state)
    return result.dynamicsbias
end

function dynamics_bias_h(m::Kuka,qk,qn,h::Vector{T}) where T
    state = m.state_cache1[T]
    result = m.result_cache1[T]
    set_configuration!(state,qk)
    set_velocity!(state,(qn-qk)/h[1])
    dynamics_bias!(result,state)
    return result.dynamicsbias
end

function B_func(m::Kuka,q)
    Diagonal(ones(m.nx))
end

function discrete_dynamics(model,x1,x2,x3,u_ctrl,h,t)
    ((1/h[1])*(M_func(model,x1)*(x2 - x1)
               - M_func(model,x2)*(x3 - x2))
     + h[1]*0.5*dynamics_bias(model,x2,x3,h)
     + transpose(B_func(model,x3))*u_ctrl)
end

qL = -Inf*ones(nq)
qU = Inf*ones(nq)

model = Kuka(qL,qU,
    state_cache1,state_cache2,state_cache3,
    results_cache1,results_cache2,results_cache3,
    world,
    nq,nu)

q0 = rand(nq)
u0 = rand(nu)
h0 = [1.0]

# test RBD methods
using ForwardDiff, LinearAlgebra

M_func(model,q0)
M(x) = M_func(model,x)
ForwardDiff.jacobian(M,q0)

dynamics_bias_qk(model,q0,q0,h0)
dynamics_bias_qn(model,q0,q0,h0)
dynamics_bias_h(model,q0,q0,h0)
D(x) = dynamics_bias_qk(model,x,q0,h0)
ForwardDiff.jacobian(D,q0)
D(x) = dynamics_bias_qn(model,q0,x,h0)
ForwardDiff.jacobian(D,q0)
D(x) = dynamics_bias_h(model,q0,q0,x)
ForwardDiff.jacobian(D,h0)

@time discrete_dynamics(model,q0,q0,q0,u0,h0,0)
F(x) = discrete_dynamics(model,x,q0,q0,u0,h0,0)
ForwardDiff.jacobian(F,q0)
F(x) = discrete_dynamics(model,q0,x,q0,u0,h0,0)
ForwardDiff.jacobian(F,q0)
F(x) = discrete_dynamics(model,q0,q0,x,u0,h0,0)
ForwardDiff.jacobian(F,q0)
F(x) = discrete_dynamics(model,q0,q0,q0,x,h0,0)
ForwardDiff.jacobian(F,u0)
F(x) = discrete_dynamics(model,q0,q0,q0,u0,x,0)
ForwardDiff.jacobian(F,h0)
