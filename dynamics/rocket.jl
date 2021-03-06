using ForwardDiff
abstract type Rocket end

struct RocketNominal{T} <: Rocket
	g::T
	m1::T
	l1::T
	J::T

	nx::Int
	nu::Int
end

struct RocketSlosh{T} <: Rocket
	g::T
	m1::T
	l1::T
	J::T

	m2::T
	l2::T
	l3::T

	nx::Int
	nu::Int
end

g = 9.81
m1 = 1.0 # mass of rocket
l1 = 0.5 # length from COM to thruster
J = 1.0/12.0*m1*(2*l1)^2 # inertia of rocket

m2 = 0.1 # mass of pendulum
l2 = 0.1 # length from COM to pendulum
l3 = 0.1 # length of pendulum

# nominal model
nu = 2
nx_nom = 6
model_nom = RocketNominal(g,m1,l1,J,nx_nom,nu)

# slosh model
nx_slosh = 8
model_slosh = RocketSlosh(g,m1-m2,l1,J,m2,l2,l3,nx_slosh,nu)

function kinematics_thruster(model,q)
	x,z,θ = q[1:3]
	px = x + model.l1*sin(θ)
	pz = z - model.l1*cos(θ)

	return [px;pz]
end

function jacobian_thruster(model::RocketNominal,q)
	x,z,θ = q[1:3]

	return [1.0 0.0 model.l1*cos(θ);
			0.0 1.0 model.l1*sin(θ)]
end

function jacobian_thruster(model::RocketSlosh,q)
	x,z,θ = q[1:3]

	return [1.0 0.0 model.l1*cos(θ) 0.0;
			0.0 1.0 model.l1*sin(θ) 0.0]
end

function kinematics_mass(model::RocketSlosh,q)
	xp = q[1] + l2*sin(q[3]) + l3*sin(q[4])
	zp = q[2] - l2*cos(q[3]) - l3*cos(q[4])

	return [xp;zp]
end

function jacobian_mass(model::RocketSlosh,q)
	return [1.0 0.0 l2*cos(q[3]) l3*cos(q[4]);
	        0.0 1.0 l2*sin(q[3]) l3*sin(q[4])]
end

function lagrangian(model::RocketNominal,q,q̇)
	return (0.5*model.m1*(q̇[1]^2 + q̇[2]^2) + 0.5*model.J*q̇[3]^2
			- model.m1*model.g*q[2])
end

function lagrangian(model::RocketSlosh,q,q̇)
	zp = kinematics_mass(model,q)[2]
	vp = jacobian_mass(model,q)*q̇

	return (0.5*model.m1*(q̇[1]^2 + q̇[2]^2) + 0.5*model.J*q̇[3]^2
			- model.m1*model.g*q[2]
			+ 0.5*model.m2*vp'*vp
			- model.m2*model.g*zp)
end

function dLdq(model,q,q̇)
	Lq(x) = lagrangian(model,x,q̇)
	ForwardDiff.gradient(Lq,q)
end

function dLdq̇(model,q,q̇)
	Lq̇(x) = lagrangian(model,q,x)
	ForwardDiff.gradient(Lq̇,q̇)
end

function dynamics(model::Rocket,x,u)
	nq = convert(Int,floor(model.nx/2))
	q = x[1:nq]
	q̇ = x[nq .+ (1:nq)]
	tmp_q(z) = dLdq̇(model,z,q̇)
	tmp_q̇(z) = dLdq̇(model,q,z)
	[q̇;
	 ForwardDiff.jacobian(tmp_q̇,q̇)\(-1.0*ForwardDiff.jacobian(tmp_q,q)*q̇
	 	+ dLdq(model,q,q̇)
		+ jacobian_thruster(model,q)'*u)]
end

function policy(model::RocketSlosh,K,x,u,x_nom,u_nom)
	u_nom - reshape(K,model.nu,model.nx-2)*(output(model,x) - x_nom)
end

function output(model::RocketSlosh,x)
	x[[(1:3)...,(5:7)...]]
end

function output(model::RocketNominal,x)
	x
end

function visualize!(vis,model::Rocket,q;
       Δt=0.1,r_rocket=0.1,r_pad=0.25)

	# obj_path = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/rocket.obj"
	# mtl_path = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/rocket.mtl"
	#
	# ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.01)
	# setobject!(vis["rocket"],ctm)#MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
	# settransform!(vis["rocket"], LinearMap(RotZ(pi)*RotX(pi/2.0)))

	# body = Cylinder(Point3f0(0,0,-model.l1),Point3f0(0,0,model.l1),convert(Float32,r_rocket))
	# setobject!(vis["rocket"],body,MeshPhongMaterial(color=RGBA(1,1,1,1.0)))
	#
	# landing_pad = Cylinder(Point3f0(0,0,-0.1),Point3f0(0,0,0),convert(Float32,r_rocket+r_pad))
	# setobject!(vis["landing_pad"],landing_pad,MeshPhongMaterial(color=RGBA(0,0,0,1.0)))

	obj_rocket = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_booster.obj"
	mtl_rocket = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_booster.mtl"

	rkt_offset = [4.0,-6.35,0.2]
	ctm = ModifiedMeshFileObject(obj_rocket,mtl_rocket,scale=1.0)
	setobject!(vis["rocket"],ctm)
	settransform!(vis["rocket"], compose(Translation((q[T][1:3] + rkt_offset)...),LinearMap(RotZ(-pi)*RotX(pi/2.0))))

	obj_platform = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_platform.obj"
	mtl_platform = "/home/taylor/Research/direct_policy_optimization/dynamics/rocket/space_x_platform.mtl"

	ctm_platform = ModifiedMeshFileObject(obj_platform,mtl_platform,scale=1.0)
	setobject!(vis["platform"],ctm_platform)
	settransform!(vis["platform"], compose(Translation(0.0,0.0,0.0),LinearMap(RotZ(pi)*RotX(pi/2))))


   	anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q)
        MeshCat.atframe(anim,t) do
			settransform!(vis["rocket"], compose(Translation(q[t][1]+rkt_offset[1],0.0+rkt_offset[2],q[t][2]+rkt_offset[3]),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
