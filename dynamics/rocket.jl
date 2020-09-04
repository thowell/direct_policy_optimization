"""
Simple 2D rocket model
"""
mutable struct Rocket{T}
	mr::T
	Jr::T
	g::T

	l1::T
	l2::T

	nx::Int
	nu::Int
end

function dynamics(model::Rocket,x,u)
	θ = x[3]

	FE = u[1]
	FT = u[2]
	φ = u[3]

	return @SVector [x[4],
					 x[5],
					 x[6],
					 (-FT*cos(θ) - FE*cos(φ)*sin(θ) + FE*cos(θ)*sin(φ))/model.mr,
					 (-FT*sin(θ) + FE*cos(φ)*cos(θ) + FE*sin(φ)*sin(θ) - model.mr*model.g)/model.mr,
					 (FT*model.l2 - FE*sin(φ)*model.l1)/model.Jr]
end

nx = 6
nu = 3
model = Rocket(1.0,0.1,9.81,0.5,0.5,nx,nu)

"""
Simple 2D rocket model with tank slosh
"""
mutable struct RocketSlosh{T}
	mr::T
	Jr::T
	mf::T
	g::T

	l1::T
	l2::T
	l3::T

	nx::Int
	nu::Int
end

function dynamics(model::RocketSlosh,x,u)
	θ = x[3]
	ψ = x[4]
	FE = u[1]
	FT = u[2]
	φ = u[3]

	return @SVector [x[5],
					 x[6],
					 x[7],
					 x[8],
					 (-FT*cos(θ) - FE*cos(φ)*sin(θ) + FE*cos(θ)*sin(φ))/model.mr,
					 (-FT*sin(θ) + FE*cos(φ)*cos(θ) + FE*sin(φ)*sin(θ) - (model.mr + model.mf)*model.g)/model.mr,
					 (FT*model.l2 - FE*sin(φ)*model.l1 - model.mf*model.g*sin(ψ)*model.l3)/model.Jr,
					 (-model.mf*model.g*sin(ψ)*model.l3)/(model.mf*model.l3*model.l3)]
end

nx_slosh = 8
nu_slosh = 3
model_slosh = RocketSlosh(1.0,0.1,0.01,9.81,0.5,0.5,0.1,nx_slosh,nu_slosh)
