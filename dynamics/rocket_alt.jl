"""
Simple 2D rocket model
"""
mutable struct Rocket{T}
	mr::T
	lr::T
	g::T

	nx::Int
	nu::Int
end

function dynamics(model::Rocket,x,u)
	θ = x[3]

	F = u[1]
	φ = u[2]

	return @SVector [x[4],
					 x[5],
					 x[6],
					 -sin(φ)*F/model.mr,
					 (cos(φ)*F - model.mr*model.g)/model.mr,
					 sin(θ)*model.mr*model.g*model.lr/(model.mr*model.lr*model.lr)]
end

nx = 6
nu = 2
model = Rocket(1.0,1.0,9.81,nx,nu)

mutable struct RocketSlosh{T}
	mr::T
	lr::T
	g::T

	fp::T
	lf::T

	nx::Int
	nu::Int
end

function dynamics(model::RocketSlosh,x,u)
	θ = x[3]
	ψ = x[4]

	F = u[1]
	φ = u[2]

	mr = (1.0-model.fp)*model.mr
	mf = model.fp*model.mr

	return @SVector [x[5],
					 x[6],
					 x[7],
					 x[8],
					 -sin(φ)*F/model.mr,
					 (cos(φ)*F - model.mr*model.g)/model.mr,
					 (sin(θ)*mr*model.g*model.lr - sin(ψ)*mf*model.g*model.lf)/(mr*model.lr*model.lr + mf*model.lf*model.lf),
					 -sin(ψ)*mf*model.g*model.lf/(mf*model.lf*model.lf)]
end

nx_slosh = 8
nu_slosh = 2
model_slosh = RocketSlosh(1.0,1.0,9.81,0.5,0.1,nx_slosh,nu_slosh)
