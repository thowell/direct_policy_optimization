include(joinpath(pwd(),"src/direct_policy_optimization.jl"))

g = 9.81
m1 = 0.95 # mass of rocket
l1 = 1.0 # length from COM to thruster
J = 1.0/12.0*m1*(2*l1)^2 # inertia of rocket

m2 = 0.05 # mass of pendulum
l2 = 0.25 # length from COM to pendulum
l3 = 0.1 # length of pendulum

struct Rocket{T}
	g::T
	m1
	l1
	J

	m2
	l2
	l3

	nx
	nu
end

nx = 8
nu = 2
model = Rocket(g,m1,l1,J,m2,l2,l3,nx,nu)

function kinematics_thruster(q)
	x,z,θ = q[1:3]
	px = x + l1*sin(θ)
	pz = z - l1*cos(θ)

	return [px;pz]
end

function jacobian_thruster(q)
	x,z,θ = q[1:3]

	return [1.0 0.0 l1*cos(θ) 0.0;
			0.0 1.0 l1*sin(θ) 0.0]
end

function kinematics_mass(q)
	xp = q[1] + l2*sin(q[3]) + l3*sin(q[4])
	zp = q[2] - l2*cos(q[3]) - l3*cos(q[4])

	return [xp;zp]
end

function jacobian_mass(q)
	return [1.0 0.0 l2*cos(q[3]) l3*cos(q[4]);
	        0.0 1.0 l2*sin(q[3]) l3*sin(q[4])]
end

# test Jacobians
q0 = rand(4)
q̇0 = rand(4)
@assert norm(vec(ForwardDiff.jacobian(kinematics_thruster,q0))
 	- vec(jacobian_thruster(q0))) < 1.0e-8
@assert norm(vec(ForwardDiff.jacobian(kinematics_mass,q0))
 	- vec(jacobian_mass(q0))) < 1.0e-8

function lagrangian(q,q̇)
	zp = kinematics_mass(q)[2]
	vp = jacobian_mass(q)*q̇

	return (0.5*m1*(q̇[1]^2 + q̇[2]^2) + 0.5*J*q̇[3]^2
			- m1*g*q[2]
			+ 0.5*m2*vp'*vp
			- m2*g*zp)
end

lagrangian(q0,q̇0)

function dLdq(q,q̇)
	Lq(x) = lagrangian(x,q̇)
	ForwardDiff.gradient(Lq,q)
end

function dLdq̇(q,q̇)
	Lq̇(x) = lagrangian(q,x)
	ForwardDiff.gradient(Lq̇,q̇)
end

dLdq(q0,q̇0)
dLdq̇(q0,q̇0)

function dynamics(model::Rocket,x,u)
	q = x[1:4]
	q̇ = x[5:8]
	tmp_q(z) = dLdq̇(z,q̇)
	tmp_q̇(z) = dLdq̇(q,z)
	[q̇;
	 ForwardDiff.jacobian(tmp_q̇,q̇)\(-1.0*ForwardDiff.jacobian(tmp_q,q)*q̇
	 	+ dLdq(q,q̇)
		+ jacobian_thruster(q)'*u)]
end

dynamics([q0;q̇0],rand(2))
