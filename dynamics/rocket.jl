mutable struct Rocket{T}
	m::T
	J::T
	g::T

	l1::T
	l2::T
	ln::T
	ls::T
end

function dynamics(model::Rocket,x,u)
	θ = x[3]
	FE = u[1]
	FS = u[2]
	φ = u[3]

	return @SVector [x[4],
					 x[5],
					 x[6],
					 (FE*cos(φ)*sin(θ) + FE*cos(θ)*sin(φ) + FS*cos(θ))/model.m,
					 FE*cos(φ)*cos(θ) - FE*sin(φ)*sin(θ) - FS*sin(θ) - model.m*model.g,
					 (-FE*sin(φ)*(model.l1 + model.ln*cos(φ)) + model.l2*FS)/model.J]
end
"""
https://project-archive.inf.ed.ac.uk/msc/20172139/msc_proj.pdf
https://www.reddit.com/r/spacex/comments/1xhuok/legbased_stability_and_moments_of_inertia/
"""
model = Rocket(20.0e3,75.0e3,9.81,35.0,35.0,1.0,1.5)
