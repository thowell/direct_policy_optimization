using MeshCat,MeshCatMechanisms,RigidBodyDynamics
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, MeshCat

urdf_original = joinpath(pwd(),"dynamics/kuka/kuka.urdf")
urdf_new = joinpath(pwd(),"dynamics/kuka/temp/kuka.urdf")

function write_kuka_urdf()
    kuka_mesh_dir = joinpath(pwd(),"dynamics/kuka/meshes")
    temp_dir = joinpath(pwd(),"dynamics/kuka/temp")
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

urdf_path = joinpath(pwd(),"dynamics/kuka/temp/kuka.urdf")

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
q_init = zeros(nq)
q_init[1] = pi/4
q_init[2] = pi/4

q_init[4] = -pi/2
# q[6] = pi/2
# q[7] = pi/2
set_configuration!(state,q_init)
set_configuration!(mvis, q_init)
ee = findbody(kuka, "iiwa_link_7")
ee_point = Point3D(default_frame(ee), 0.0, 0.0, 0.0)
setelement!(mvis, ee_point, 0.055)

ee_jacobian_frame = ee_point.frame
ee_jacobian_path = path(kuka, root_body(kuka), ee)

world = root_frame(kuka)
ee_in_world = transform(state, ee_point, world).v
desired = Point3D(world,0.25,0.35,0.0)
q_res1 = jacobian_transpose_ik!(state,ee,ee_point,desired,
    visualize=true,mvis=mvis)
q_res1 = Array(q_res1)
set_configuration!(mvis, q_res1)
ϕ_func(model,q_res1)

desired = Point3D(world,0.75,0.85,0.0)
q_res2 = jacobian_transpose_ik!(state,ee,ee_point,desired,
    visualize=true,mvis=mvis)
q_res2 = Array(q_res2)
ϕ_func(model,q_res2)
set_configuration!(mvis, q_res2)

using StaticArrays

# Kuka iiwa arm parsed from URDF using RigidBodyDynamics.jl
mutable struct Kuka{T}
    qL::Vector{T}
    qU::Vector{T}

    state_cache1
    state_cache2
    state_cache3

    result_cache1
    result_cache2
    result_cache3

    world

	ee
    ee_point
    ee_jacobian_frame
    ee_jacobian_path

	μ

	nx
    nu
    nu_ctrl
    nc
    nf
    nb

    idx_u
    idx_λ
    idx_b
    idx_ψ
    idx_η
    idx_s

    α
end

# Dimensions
nq = 7 # configuration dim
nu_ctrl = 7 # control dim
nc = 1 # number of contact points
nf = 4 # number of faces for friction cone
nb = nc*nf

nx = nq
nu_contact = nc + nb + nc + nb + 1
nu = nu_ctrl + nu_contact

idx_u = (1:nu_ctrl)
idx_λ = nu_ctrl .+ (1:nc)
idx_b = nu_ctrl + nc .+ (1:nb)
idx_ψ = nu_ctrl + nc + nb .+ (1:nc)
idx_η = nu_ctrl + nc + nb + nc .+ (1:nb)
idx_s = nu_ctrl + nc + nb + nc + nb + 1

α_kuka = 1.0
μ_kuka = 0.1

results_cache1 = DynamicsResultCache(kuka)
results_cache2 = DynamicsResultCache(kuka)
results_cache3 = DynamicsResultCache(kuka)

state_cache1 = StateCache(kuka)
state_cache2 = StateCache(kuka)
state_cache3 = StateCache(kuka)

# Methods
function M_func(m::Kuka,q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state,q)
    mass_matrix!(result.massmatrix,state)
    return result.massmatrix
end

function C_func(m::Kuka,qk,qn,h)
    if eltype(qk) <: ForwardDiff.Dual
		return dynamics_bias_qk(m,qk,qn,h)
	elseif eltype(qn) <: ForwardDiff.Dual
		return dynamics_bias_qn(m,qk,qn,h)
	elseif eltype(h) <: ForwardDiff.Dual
		return dynamics_bias_h(m,qk,qn,h)
	else
		return _dynamics_bias(m,qk,qn,h)
	end
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

function friction_cone(model,u)
    @SVector [model.μ*u[model.idx_λ[1]] - sum(u[model.idx_b])]
end

function maximum_energy_dissipation(model,x2,x3,u,h)
    ψ_stack = u[model.idx_ψ][1]*ones(model.nb)
    η = u[model.idx_η]
    P_func(model,x3)*(x3-x2)/h[1] + ψ_stack - η
end

function discrete_dynamics(model::Kuka,x1,x2,x3,u,h,t)
	u_ctrl = u[model.idx_u]
	λ = u[model.idx_λ]
	b = u[model.idx_b]

    ((1/h[1])*(M_func(model,x1)*(x2 - x1)
               - M_func(model,x2)*(x3 - x2))
     - h[1]*C_func(model,x2,x3,h)
     + transpose(B_func(model,x3))*u_ctrl
	 + transpose(N_func(model,x3))*λ
	 + transpose(P_func(model,x3))*b)
end

function kinematics_ee(m::Kuka,q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    set_configuration!(state,q)
    return transform(state, m.ee, m.world).v
end

function ϕ_func(m::Kuka,q::Vector{T}) where T
    state = m.state_cache3[T]
    set_configuration!(state,q)
    @SVector [transform(state, m.ee_point, m.world).v[3]]
end

function N_func(m::Kuka,q::AbstractVector{T}) where T
    state = m.state_cache3[T]

    # N = zeros(T,m.nc,m.nq)

    set_configuration!(state,q)

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T,3,m.nx))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path,ee_in_world) #TODO confirm this is correct

    # N[1,:] = pj1.J[3,:]
    # N[2,:] = pj2.J[3,:]

    return pj1.J[3:3,:]
end

function P_func(m::Kuka,q::AbstractVector{T}) where T
    state = m.state_cache3[T]

	map = @SMatrix [1 0;
				    -1 0;
				    0 1;
				    0 -1]
    # P = zeros(T,m.nb,m.nq)

    set_configuration!(state,q)

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T,3,m.nx))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path, ee_in_world) #TODO confirm this is correct

    return map*pj1.J[1:2,:]
end

qL = -Inf*ones(nq)
qU = Inf*ones(nq)

model = Kuka(qL,qU,
    state_cache1,state_cache2,state_cache3,
    results_cache1,results_cache2,results_cache3,
    world,
	ee,
	ee_point,
	ee_jacobian_frame,
	ee_jacobian_path,
	μ_kuka,
	nx,nu,nu_ctrl,
	nc,nf,nb,
	idx_u,
	idx_λ,
	idx_b,
	idx_ψ,
	idx_η,
	idx_s,
	α_kuka)

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

ϕ_func(model,q_res)
N_func(model,q0)
P_func(model,q0)

state = model.state_cache3[Float64]
set_configuration!(state,q0)
set_configuration!(mvis,q0)
@SVector [transform(state, model.ee_point, model.world).v[3]]

include("/home/taylor/Research/direct_policy_optimization/src/sample_trajectory_optimization.jl")
using Plots

# Horizon
T = 20

tf = 1.0
Δt = tf/T


# Bounds
# xl <= x <= xu
xu_traj = [model.qU for t=1:T]
xl_traj = [model.qL for t=1:T]

xl_traj[1] = q_res1
xu_traj[1] = q_res1
# # # xu_traj[T] = x1
# #
xl_traj[2] = q_res1
xu_traj[2] = q_res1
# xl_traj[T] = x1

# xu_traj = [q_res for t=1:T]
# xl_traj = [q_res for t=1:T]

# ul <= u <= uu
uu = Inf*ones(model.nu)
ul = -Inf*ones(model.nu)
ul[nu_ctrl .+ (1:nu_contact)] .= 0.0

# uu[1:nu_ctrl] .= 0.0
# ul[1:nu_ctrl] .= 0.0

ul_traj = [ul for t = 1:T-2]
uu_traj = [uu for t = 1:T-2]

# h = h0 (fixed timestep)
hu = Δt
hl = Δt

# Objective
Q = [t<T ? Diagonal(1.0*ones(model.nx)) : Diagonal(1.0*ones(model.nx)) for t = 1:T]
R = [Diagonal(1.0e-2*ones(model.nu_ctrl)) for t = 1:T-2]
c = 0.0
x_ref = [q_res1,linear_interp(q_res1,q_res2,T-1)...]

set_configuration!(state,q_res2)
u_ref = Δt*Array(RigidBodyDynamics.dynamics_bias(state))

obj = QuadraticTrackingObjective(Q,R,c,
    [q_res2 for t=1:T],[u_ref for t=1:T-2])
model.α = 10.0
penalty_obj = PenaltyObjective(model.α)
multi_obj = MultiObjective([obj,penalty_obj])

# Problem
prob = init_problem(model.nx,model.nu,T,model,multi_obj,
                    xl=xl_traj,
                    xu=xu_traj,
                    ul=ul_traj,
                    uu=uu_traj,
                    hl=[hl for t = 1:T-2],
                    hu=[hu for t = 1:T-2],
                    )

# MathOptInterface problem
prob_moi = init_MOI_Problem(prob)

# Trajectory initialization
X0 = deepcopy(x_ref)
U0 = [[u_ref;1.0e-5*rand(nu_contact)] for t = 1:T-2] # random controls

# Pack trajectories into vector
Z0 = pack(X0,U0,Δt,prob)
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:ipopt)
# @time Z_nominal = solve(prob_moi,copy(Z_nominal),nlp=:SNOPT)

X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

[ϕ_func(model,X_nom[t]) for t = 1:T]
x_nom = [X_nom[t][1] for t = 1:T]
z_nom = [X_nom[t][2] for t = 1:T]
u_nom = [U_nom[t][model.idx_u] for t = 1:T-2]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
b_nom = [U_nom[t][model.idx_b] for t = 1:T-2]
ψ_nom = [U_nom[t][model.idx_ψ[1]] for t = 1:T-2]
η_nom = [U_nom[t][model.idx_η] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)

plot(hcat(u_nom...)')
plot(λ_nom,linetype=:steppost)
plot(hcat(b_nom...)',linetype=:steppost)
plot(ψ_nom,linetype=:steppost)
plot(hcat(η_nom...)',linetype=:steppost)
plot(hcat(U_nom...)',linetype=:steppost)

ts = range(0,stop=H_nom[1]*T,length=T)

animation = MeshCat.Animation(mvis,ts,X_nom)
setanimation!(mvis,animation)
