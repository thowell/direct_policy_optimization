include(joinpath(pwd(),"src/direct_policy_optimization.jl"))

g = 9.81
m1 = 1.0
l1 = 1.0
lc1 = 0.5
I1 = 1.0

m2 = 1.0
l2 = 1.0
lc2 = 0.5
I2 = 1.0

function lagrangian(q,q̇)
	θ1, θ2 = q[1:2]
	dθ1, dθ2 = q̇[1:2]
	(
	0.5*I1*dθ1^2
	+0.5*(m2*l1^2 + I2 + 2*m2*l1*lc2*cos(θ2))*dθ1^2
	+ 0.5*I2*dθ2^2
	+ (I2 + m2*l1*lc2*cos(θ2))*dθ1*dθ2
	- m1*g*lc1*cos(θ1)
	- m2*g*(l1*cos(θ1) + lc2*cos(θ1+θ2))
	)
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
	 -ForwardDiff.jacobian(tmp_q̇,q̇)\(ForwardDiff.jacobian(tmp_q,q)*q̇
	 	- dLdq(q,q̇)
		- [0; 1.]*u)]
end

dLdq(q0,q̇0)

τ = -1.0*[
	-m1*g*lc1*sin(q0[1]) - m2*g*(l1*sin(q0[1]) + lc2*sin(q0[1] + q0[2]));
	-m2*g*lc2*sin(q0[1] + q0[2])
]

# tmp(z) = dLdq̇(q0,z)
# ForwardDiff.jacobian(tmp,q̇0)
#
# M = [(I1 + I2 + m2*l1^2 + 2*m2*l1*lc2*cos(q0[2])) (I2 + m2*l1*lc2*cos(q0[2]));
# 	(I2 + m2*l1*lc2*cos(q0[2])) I2]
