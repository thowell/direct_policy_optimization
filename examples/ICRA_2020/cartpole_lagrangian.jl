include(joinpath(pwd(),"src/direct_policy_optimization.jl"))

g = 9.81
mc = 1.2
mp = 4.5
l = .4

function kinematics_cart(q)
	x,θ = q[1:2]
	px = x
	pz = 0

	return [px;pz]
end

function jacobian_cart(q)
	x,θ = q[1:2]

	return [1.0 0.0]
end

function kinematics_mass(q)
	x,θ = q[1:2]

	xp = x + l*sin(θ)
	zp = -l*cos(θ)

	return [xp;zp]
end

function jacobian_mass(q)
	x,θ = q[1:2]
	return [1.0 l*cos(θ);
	        0.0 l*sin(θ)]
end

# test Jacobians
q0 = rand(2)
q̇0 = rand(2)
@assert norm(vec(ForwardDiff.jacobian(kinematics_cart,q0))
 	- vec(jacobian_cart(q0))) < 1.0e-8
@assert norm(vec(ForwardDiff.jacobian(kinematics_mass,q0))
 	- vec(jacobian_mass(q0))) < 1.0e-8

function lagrangian(q,q̇)
	x,θ = q[1:2]
	ẋ,θ̇ = q̇[1:2]

	vp = jacobian_mass(q)*q̇

	return 0.5*vp'*Diagonal([mc;mp])*vp - mp*g*l*cos(θ)
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

function dynamics(x,u)
	q = x[1:2]
	q̇ = x[3:4]
	tmp_q(z) = dLdq̇(z,q̇)
	tmp_q̇(z) = dLdq̇(q,z)
	[q̇;
	 -ForwardDiff.jacobian(tmp_q̇,q̇)\(ForwardDiff.jacobian(tmp_q̇,q)*q̇
	 	- dLdq(q,q̇)
		- jacobian_cart(q)'*u)]
end

u0 = rand(1)
dynamics([q0;q̇0],u0)[3:4]

[1/(mc+mp*sin(q0[2])^2)*(u0[1] + mp*sin(q0[2])*(l*q̇0[2]^2 + g*cos(q0[2])));
 1/(l*(mc+mp*sin(q0[2])^2))*(-u0[1]*cos(q0[2]) - mp*l*(q̇0[2]^2)*cos(q0[2])*sin(q0[2]) - (mc + mp)*g*sin(q0[2]))]
