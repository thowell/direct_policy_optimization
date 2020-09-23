using MeshCat,MeshCatMechanisms,RigidBodyDynamics
using FileIO, MeshIO, GeometryTypes, CoordinateTransformations, MeshCat

t = range(-1,stop=3,length=100)
k = 10.0
shift = 1.5
softplus(x) = log(1 + exp(k*(x-shift)))/k
dsoftplus(x) = 1.0/(1.0 + exp(-k*(x-shift)))
# ForwardDiff.derivative(softplus,1.2) - dsoftplus(1.2)
plot(t,softplus.(t))

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

# course
include("../golf_particle.jl")
m = model
r_ball = 0.1
f = x -> x[3] - softplus(x[1])

sdf = SignedDistanceField(f, HyperRectangle(Vec(-5, -10, -1), Vec(10, 10, 4)))
mesh = HomogenousMesh(sdf, MarchingTetrahedra())
setobject!(vis["slope"], mesh,
		   MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))
settransform!(vis["slope"], compose(Translation(0.0,5.0,0.0)))
#
circle1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.25),convert(Float32,1.0-r_ball))
	setobject!(vis["circle1"],circle1,
	MeshPhongMaterial(color=RGBA(86/255,125/255,20/255,1.0)))
settransform!(vis["circle1"], compose(Translation(0.25,-1.5,0.0)))

# circle2 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.25),convert(Float32,m.rx1*0.5))
# 	setobject!(vis["circle2"],circle2,
# 	MeshPhongMaterial(color=RGBA(0,0,0,0.5)))

setobject!(vis["ball"], HyperSphere(Point3f0(0),
			convert(Float32,r_ball)),
			MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

settransform!(vis["ball"], compose(Translation(0.0,2.0,0.05)))

hole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.01),convert(Float32,r_ball))
	setobject!(vis["hole"],hole,
	MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
settransform!(vis["hole"], compose(Translation(0.66,-3.0,0.0)))

nq = num_positions(kuka)
q_init = zeros(nq)
q_init[1] = 0
q_init[2] = pi/4

q_init[4] = -pi/2
q_init[5] = 0
q_init[6] = -pi/4
q_init[7] = 0
set_configuration!(state,q_init)
set_configuration!(mvis, q_init)
ee = findbody(kuka, "iiwa_link_7")
ee_point = Point3D(default_frame(ee), 0.374, 0.0, 0.0)
setelement!(mvis, ee_point, 0.01)
ee_jacobian_frame = ee_point.frame
ee_jacobian_path = path(kuka, root_body(kuka), ee)

world = root_frame(kuka)
ee_in_world = transform(state, ee_point, world).v
desired = Point3D(world,0.5,0.0,0.0)
q_res1 = jacobian_transpose_ik!(state,ee,ee_point,desired,
    visualize=true,mvis=mvis)
q_res1 = Array(q_res1)
set_configuration!(mvis, q_res1)
set_configuration!(state,q_res1)
ee_in_world = transform(state, ee_point, world).v

desired = Point3D(world,0.75,0.0,0.0)
q_res2 = jacobian_transpose_ik!(state,ee,ee_point,desired,
    visualize=true,mvis=mvis)
q_res2 = Array(q_res2)
set_configuration!(mvis, q_res2)
set_configuration!(state,q_res2)
ee_in_world = transform(state, ee_point, world).v

using StaticArrays

# Kuka iiwa arm parsed from URDF using RigidBodyDynamics.jl
mutable struct KukaParticle{T}
	mp::T
	rp::T

	# course parameters
	rx1
	ry1

	rx2
	ry2

	a_exp
	shift_exp

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

	μ_ee_p
	μ1
	μ2
	μ3

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
nq = 7 + 3 # configuration dim
nu_ctrl = 7 # control dim
nc = 3 # number of contact points
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

mp = 0.05
rp = 0.05
α_kuka = 1.0
μ_ee_p = 1.0
μ1 = 0.01
μ2 = 0.1
μ3 = 0.1


results_cache1 = DynamicsResultCache(kuka)
results_cache2 = DynamicsResultCache(kuka)
results_cache3 = DynamicsResultCache(kuka)

state_cache1 = StateCache(kuka)
state_cache2 = StateCache(kuka)
state_cache3 = StateCache(kuka)

function kuka_q(q)
	q[1:7]
end

function particle_q(q)
	q[8:10]
end

# Methods
function M_func(m::KukaParticle,q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state,kuka_q(q))
    mass_matrix!(result.massmatrix,state)
    return cat(result.massmatrix,Diagonal(m.mp*ones(3)),dims=(1,2))
end

function C_func(m::KukaParticle,qk,qn,h)
    if eltype(kuka_q(qk)) <: ForwardDiff.Dual
		return [dynamics_bias_qk(m,kuka_q(qk),kuka_q(qn),h)...,0,0,m.mp*9.81]
	elseif eltype(kuka_q(qn)) <: ForwardDiff.Dual
		return [dynamics_bias_qn(m,kuka_q(qk),kuka_q(qn),h)...,0,0,m.mp*9.81]
	elseif eltype(h) <: ForwardDiff.Dual
		return [dynamics_bias_h(m,kuka_q(qk),kuka_q(qn),h)...,0,0,m.mp*9.81]
	else
		return [_dynamics_bias(m,kuka_q(qk),kuka_q(qn),h)...,0,0,m.mp*9.81]
	end
end

function _dynamics_bias(m::KukaParticle,qk,qn,h)
	T = eltype(m.qL)
	state = m.state_cache1[T]
	result = m.result_cache1[T]
	set_configuration!(state,qk)
	set_velocity!(state,(qn-qk)/h[1])
	dynamics_bias!(result,state)
    return result.dynamicsbias
end

function dynamics_bias_qk(m::KukaParticle,qk::Vector{T},qn,h) where T
    state = m.state_cache2[T]
    result = m.result_cache2[T]
    set_configuration!(state,qk)
    set_velocity!(state,(qn-qk)/h[1])
    dynamics_bias!(result,state)
    return result.dynamicsbias
end

function dynamics_bias_qn(m::KukaParticle,qk,qn::Vector{T},h) where T
    state = m.state_cache3[T]
    result = m.result_cache3[T]
    set_configuration!(state,qk)
    set_velocity!(state,(qn-qk)/h[1])
    dynamics_bias!(result,state)
    return result.dynamicsbias
end

function dynamics_bias_h(m::KukaParticle,qk,qn,h::Vector{T}) where T
    state = m.state_cache1[T]
    result = m.result_cache1[T]
    set_configuration!(state,qk)
    set_velocity!(state,(qn-qk)/h[1])
    dynamics_bias!(result,state)
    return result.dynamicsbias
end

function B_func(m::KukaParticle,q)
    [Diagonal(ones(7));zeros(3,7)]
end

function friction_cone(model::KukaParticle,u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]

    @SVector [model.μ_ee_p*λ[1] - sum(b[1:4]),#,
			  model.μ1*λ[2] - sum(b[5:8]),
			  model.μ2*λ[3] - sum(b[9:12])]
			  # model.μ3*λ[3] - sum(b[9:12])]
end

function maximum_energy_dissipation(model::KukaParticle,x2,x3,u,h)
	ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1]*ones(4);ψ[2]*ones(4);ψ[3]*ones(4)]#;ψ[4]*ones(4)]
    η = u[model.idx_η]
    P_func(model,x3)*(x3-x2)/h[1] + ψ_stack - η
end

function discrete_dynamics(model::KukaParticle,x1,x2,x3,u,h,t)
	u_ctrl = u[model.idx_u]
	λ = u[model.idx_λ]
	b = u[model.idx_b]

    ((1/h[1])*(M_func(model,x1)*(x2 - x1)
               - M_func(model,x2)*(x3 - x2))
     - h[1]*C_func(model,x2,x3,h)
     + B_func(model,x3)*u_ctrl
	 + transpose(N_func(model,x3))*λ
	 + transpose(P_func(model,x3))*b)
end

function kinematics_ee(m::KukaParticle,q::AbstractVector{T}) where T
    state = m.state_cache3[T]
    set_configuration!(state,kuka_q(q))
    return transform(state, m.ee_point, m.world).v
end

function ϕ_func(m::KukaParticle,q::Vector{T}) where T
    # state = m.state_cache3[T]
    # set_configuration!(state,kuka_q(q))
	# p_ee = transform(state, m.ee_point, m.world).v

	p_ee = kinematics_ee(m,q)
	p_p = q[8:10]
	diff = (p_ee - p_p)
	d_ee_p = diff'*diff
    @SVector [d_ee_p - m.rp^2,#,
		      ellipsoid(p_p[1],p_p[2],0.25,-1.5,1.0,1.0),
	          # # -1.0*ellipsoid(p_p[1],p_p[2],m.rx2,m.ry2),
	          p_p[3] - softplus(p_p[1])]
end

function N_func(m::KukaParticle,q::AbstractVector{T}) where T
    state = m.state_cache3[T]

    # N = zeros(T,m.nc,m.nq)

    set_configuration!(state,kuka_q(q))

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T,3,7))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path,ee_in_world) #TODO confirm this is correct

    # N[1,:] = pj1.J[3,:]
    # N[2,:] = pj2.J[3,:]

	p_ee = ee_in_world.v
	p_p = q[8:10]
	diff = (p_ee - p_p)
	# d_ee_p = diff'*diff

    return [2.0*diff'*[pj1.J[1:3,:] -Diagonal(ones(3))];#;
			zeros(1,7) transpose(∇ellipsoid(p_p[1],p_p[2],0.25,-1.5,1.0,1.0));
			# # zeros(1,7) -1.0*transpose(∇ellipsoid(p_p[1],p_p[2],m.rx2,m.ry2));
			zeros(1,7) -dsoftplus(p_p[1]) 0.0 1.0]
	# ϕ_tmp(y) = ϕ_func(model,y)
	# ForwardDiff.jacobian(ϕ_tmp,q)
end

# tmp(x) = ϕ_func(model,x)
# vec(ForwardDiff.jacobian(tmp,x1)) - vec(N_func(model,x1))

function P_func(m::KukaParticle,q::AbstractVector{T}) where T
    state = m.state_cache3[T]

	map = @SMatrix [1 0;
				    -1 0;
				    0 1;
				    0 -1]

    set_configuration!(state,kuka_q(q))

    pj1 = PointJacobian(transform(state, m.ee_point, m.world).frame,
		zeros(T,3,7))

    ee_in_world = transform(state, m.ee_point, m.world)

    point_jacobian!(pj1, state, m.ee_jacobian_path, ee_in_world) #TODO confirm this is correct

	y0 = [0.0; 1.0; 0.0]
	z0 = [0.0; 0.0; 1.0]
	p_p = particle_q(q)
    return [map*pj1.J[2:3,:] zeros(4,3);
			zeros(4,7) map*[transpose(z0);transpose(cross(z0,∇ellipsoid(p_p[1],p_p[2],0.25,-1.5,1.0,1.0)))];
			# # zeros(4,7) map*[transpose(z0); transpose(cross(z0,∇ellipsoid(p_p[1],p_p[2],m.rx2,m.ry2)))];
			zeros(4,7) map*[transpose(y0);transpose(cross(y0,[-dsoftplus(p_p[1]); 0.0; 1.0]))]]

end

qL = -Inf*ones(nx)
qU = Inf*ones(nx)

model = KukaParticle(
	mp,rp,
	rx1,ry1,
	rx2,ry2,
	a_exp,shift_exp,
	qL,qU,
    state_cache1,state_cache2,state_cache3,
    results_cache1,results_cache2,results_cache3,
    world,
	ee,
	ee_point,
	ee_jacobian_frame,
	ee_jacobian_path,
	μ_ee_p,
	μ1,μ2,μ3,
	nx,nu,nu_ctrl,
	nc,nf,nb,
	idx_u,
	idx_λ,
	idx_b,
	idx_ψ,
	idx_η,
	idx_s,
	α_kuka)

q0 = rand(nx)
u0 = rand(nu)
h0 = [1.0]

# test RBD methods
using ForwardDiff, LinearAlgebra

Array(M_func(model,q0))
M(x) = M_func(model,x)
ForwardDiff.jacobian(M,q0)

Array(P_func(model,q0))
P(x) = P_func(model,x)
ForwardDiff.jacobian(P,q0)

Array(N_func(model,q0))
N(x) = N_func(model,x)
ForwardDiff.jacobian(N,q0)

# dynamics_bias_qk(model,q0,q0,h0)
# dynamics_bias_qn(model,q0,q0,h0)
# dynamics_bias_h(model,q0,q0,h0)
# D(x) = dynamics_bias_qk(model,x,q0,h0)
# ForwardDiff.jacobian(D,q0)
# D(x) = dynamics_bias_qn(model,q0,x,h0)
# ForwardDiff.jacobian(D,q0)
# D(x) = dynamics_bias_h(model,q0,q0,x)
# ForwardDiff.jacobian(D,h0)

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

ϕ_func(model,q0)
N_func(model,q0)
P_func(model,q0)

state = model.state_cache3[Float64]
set_configuration!(state,kuka_q(q0))
set_configuration!(mvis,kuka_q(q0))
@SVector [transform(state, model.ee_point, model.world).v[3]]

include("/home/taylor/Research/direct_policy_optimization/src/sample_trajectory_optimization.jl")
using Plots

# Horizon
T = 40
# T_half = convert(Int,floor(T/2))
# T_tenth = convert(Int,floor(T/10))


tf = 2.0
Δt = tf/T

px_init = [0.66; 0.0; softplus(0.66)]
px_goal = [0.66; -3.0; softplus(0.66)]

# q1 = copy(q_init)
# q1[7] = -pi/6
# qT = copy(q_init)
# qT[7] = 0.0
x1 = [q_res1;px_init]
xT = [q_res2;px_goal]

# Bounds
# xl <= x <= xu
xu_traj = [model.qU for t=1:T]
xl_traj = [model.qL for t=1:T]

xl_traj[1] = copy(x1)
xu_traj[1] = copy(x1)
# # # xu_traj[T] = x1
# #
xl_traj[2] = copy(x1)
xu_traj[2] = copy(x1)
# xl_traj[T] = x1

# xu_traj = [q_res for t=1:T]
# xl_traj = [q_res for t=1:T]
ϕ_func(model,[q_init;px_init])
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
Q = [t<T ? Diagonal([10.0*ones(7);1.0*ones(3)]) : Diagonal([10.0*ones(7);1000.0*ones(3)]) for t = 1:T]
R = [Diagonal(1.0e-1*ones(model.nu_ctrl)) for t = 1:T-2]
c = 0.0
# x_kuka = [kuka_q(x1),linear_interp(kuka_q(x1),kuka_q(xT),T_tenth)...,[kuka_q(xT) for t = 1:(T-T_tenth)-1]...]
# x_ball = [particle_q(xT) for t = 1:T]
# #
# px2_ref = range(2.0,stop=-2.0,length=T - T_tenth)
# px1_ref = get_y.(px2_ref,2.0,2.0)
# px3_ref = exp.(model.a_exp.*(px1_ref .- model.shift_exp))
#
# px_ref = [[px1_ref[t];px2_ref[t];px3_ref[t]] for t = 1:T-T_tenth]
#
# for t = T_tenth+1:T
# 	x_ball[t] = px_ref[t-T_tenth]
# end

x_ref = linear_interp(x1,xT,T)

set_configuration!(mvis,q_init)
set_configuration!(state,q_init)
u_ref = Δt*Array(RigidBodyDynamics.dynamics_bias(state))

obj = QuadraticTrackingObjective(Q,R,c,
    [x_ref[t] for t=1:T],[u_ref for t=1:T-2])
model.α = 1000.0
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
@time Z_nominal = solve(prob_moi,copy(Z0),nlp=:ipopt,
	c_tol=1.0e-2,tol=1.0e-2,max_iter=1000,time_limit=60*5)
# @time Z_nominal = solve(prob_moi,copy(Z_nominal),nlp=:SNOPT)
X_nom, U_nom, H_nom = unpack(Z_nominal,prob)

[ϕ_func(model,X_nom[t]) for t = 1:T]
# x_nom = [X_nom[t][1] for t = 1:T]
# z_nom = [X_nom[t][2] for t = 1:T]
# u_nom = [U_nom[t][model.idx_u] for t = 1:T-2]
λ_nom = [U_nom[t][model.idx_λ[1]] for t = 1:T-2]
# b_nom = [U_nom[t][model.idx_b] for t = 1:T-2]
# ψ_nom = [U_nom[t][model.idx_ψ[1]] for t = 1:T-2]
# η_nom = [U_nom[t][model.idx_η] for t = 1:T-2]
s_nom = [U_nom[t][model.idx_s] for t = 1:T-2]
@show sum(s_nom)
#
# plot(hcat(u_nom...)')
plot(hcat(λ_nom...)',linetype=:steppost)
# plot(hcat(b_nom...)',linetype=:steppost)
# plot(hcat(ψ_nom...)',linetype=:steppost)
# plot(hcat(η_nom...)',linetype=:steppost)
# plot(hcat(U_nom...)',linetype=:steppost)


function cable_transform(y,z)
    v1 = [0,0,1]
    v2 = y[1:3,1] - z[1:3,1]
    normalize!(v2)
    ax = cross(v1,v2)
    ang = acos(v1'v2)
    R = AngleAxis(ang,ax...)

    if any(isnan.(R))
        R = I
    else
        nothing
    end

    compose(Translation(z),LinearMap(R))
end

# Visualization
function visualize!(mvis,model::KukaParticle,q;
		verbose=false,r_ball=0.1,Δt=0.1)

	p_hole = [0.66; -3.0; 0.0]
	f = x -> x[3] - softplus(x[1])

	sdf = SignedDistanceField(f, HyperRectangle(Vec(-5, -10, -1), Vec(10, 10, 4)))
	mesh = HomogenousMesh(sdf, MarchingTetrahedra())
	setobject!(vis["slope"], mesh,
			   MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))
	settransform!(vis["slope"], compose(Translation(0.0,5.0,0.0)))

	circle1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.25),convert(Float32,1.0-r_ball))
		setobject!(vis["circle1"],circle1,
		MeshPhongMaterial(color=RGBA(86/255,125/255,20/255,1.0)))
	settransform!(vis["circle1"], compose(Translation(0.25,-1.5,0.0)))


	setobject!(vis["ball"], HyperSphere(Point3f0(0),
				convert(Float32,r_ball)),
				MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

	settransform!(vis["ball"], compose(Translation(0.66,3.0,0.0)))

	hole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.01),convert(Float32,r_ball*1.5))
		setobject!(vis["hole"],hole,
		MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
	settransform!(vis["hole"], compose(Translation(p_hole...)))

	club = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.374),convert(Float32,0.025))
   	setobject!(vis["club"],club,MeshPhongMaterial(color=RGBA(0,0,0,1.0)))

	#
	state = model.state_cache1[Float64]

	ee = findbody(kuka, "iiwa_link_7")
	ee_body = Point3D(default_frame(ee), 0.0, 0.0, 0.0)
	ee_end = Point3D(default_frame(ee), 0.374, 0.0, 0.0)
	ee_body_jacobian_frame = ee_body.frame
	ee_body_jacobian_path = path(kuka, root_body(kuka), ee)
	ee_end_jacobian_frame = ee_end.frame
	ee_end_jacobian_path = path(kuka, root_body(kuka), ee)

	world = root_frame(kuka)
	ee_body_in_world = transform(state, ee_body, world).v
	ee_end_in_world = transform(state, ee_end, world).v

	anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

	T = length(q)
    for t = 1:T
        q_kuka = kuka_q(q[t])
		q_particle = particle_q(q[t])
		set_configuration!(state,kuka_q(q[t]))
		ee_body_in_world = transform(state, ee_body, world).v
		ee_end_in_world = transform(state, ee_end, world).v

        MeshCat.atframe(anim,t) do
			set_configuration!(mvis,q_kuka)
            settransform!(vis["ball"], compose(Translation(q_particle + [0.0;0.0;0.5*r_ball]),LinearMap(RotZ(0))))
			settransform!(vis["club"], cable_transform(ee_body_in_world,ee_end_in_world))

			if norm(particle_q(q[t])[1:2] - p_hole[1:2]) < 1.0e-2
				setvisible!(vis["ball"],false)
			else
				setvisible!(vis["ball"],true)
			end
		end
    end
    MeshCat.setanimation!(vis,anim)
end

visualize!(mvis,model,X_nom,Δt=Δt,r_ball=rp)
