using LinearAlgebra, Plots, ForwardDiff
using StaticArrays

# Mini-golf particle
mutable struct GolfParticle{T}
    m::T
    μ1::T # ground
	μ2::T # inner wall
	μ3::T # outer wall

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

	# course parameters
	rx1
	ry1

	rx2
	ry2

	a_exp
	shift_exp
end

# Methods
function M_func(p::GolfParticle,q)
    @SMatrix [p.m 0. 0.;
              0. p.m 0.;
              0. 0. p.m]
end

G_func(p::GolfParticle,q) = @SVector [0., 0., p.m*9.8]

C_func(::GolfParticle,qk,qn) = @SVector zeros(nq)

function ϕ_func(m::GolfParticle,x)
	[ellipsoid(x[1],x[2],m.rx1,m.ry1);
	 -1.0*ellipsoid(x[1],x[2],m.rx2,m.ry2);
	 x[3] - exp(m.a_exp*(x[1]-m.shift_exp))]
end

B_func(::GolfParticle,q) = @SMatrix [1. 0. 0.;
	                                 0. 1. 0.;
	                                 0. 0. 1.]

function N_func(m::GolfParticle,x)
	[transpose(∇ellipsoid(x[1],x[2],m.rx1,m.ry1));
	 -1.0*transpose(∇ellipsoid(x[1],x[2],m.rx2,m.ry2));
	 -m.a_exp*exp(m.a_exp*(x[1]-m.shift_exp)) 0.0 1.0]
end

function P_func(m::GolfParticle,x)
	y0 = [0.0; 1.0; 0.0]
	z0 = [0.0; 0.0; 1.0]

	[transpose(z0);
	 -1.0*transpose(z0);
	 transpose(cross(z0,∇ellipsoid(x[1],x[2],m.rx1,m.ry1)));
	 -1.0*transpose(cross(z0,∇ellipsoid(x[1],x[2],m.rx1,m.ry1)));
	 transpose(z0);
	 -1.0*transpose(z0);
	 transpose(cross(z0,∇ellipsoid(x[1],x[2],m.rx2,m.ry2)));
	 -1.0*transpose(cross(z0,∇ellipsoid(x[1],x[2],m.rx2,m.ry2)));
	 transpose(y0);
	 -1.0*transpose(y0);
	 transpose(cross(y0,[-m.a_exp*exp(m.a_exp*(x[1]-m.shift_exp)); 0.0; 1.0]));
	 -1.0*transpose(cross(y0,[-m.a_exp*exp(m.a_exp*(x[1]-m.shift_exp)); 0.0; 1.0]))]
end

function friction_cone(model,u)
	λ = u[model.idx_λ]
	b = u[model.idx_b]

    @SVector [model.μ1*λ[1] - sum(b[1:4]),
		      model.μ2*λ[2] - sum(b[5:8]),
		      model.μ3*λ[3] - sum(b[9:12])]
end

function maximum_energy_dissipation(model,x2,x3,u,h)
	ψ = u[model.idx_ψ]
    ψ_stack = [ψ[1]*ones(4); ψ[2]*ones(4); ψ[3]*ones(4)]
    η = u[model.idx_η]
    P_func(model,x3)*(x3-x2)/h[1] + ψ_stack - η
end

function visualize!(vis,m::GolfParticle,q;
	Δt=0.1,r_ball=0.1)

	f = x -> x[3] - exp(m.a_exp*(x[1]-m.shift_exp))

	sdf = SignedDistanceField(f, HyperRectangle(Vec(-3, -6, -1), Vec(10, 6, 4)))
	mesh = HomogenousMesh(sdf, MarchingTetrahedra())
	setobject!(vis["slope"], mesh,
	           MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))
	settransform!(vis["slope"], compose(Translation(0.0,3.0,0.0)))

	circle1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.35),convert(Float32,m.rx1-r_ball))
		setobject!(vis["circle1"],circle1,
		MeshPhongMaterial(color=RGBA(86/255,125/255,20/255,1.0)))

	# circle2 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.25),convert(Float32,m.rx1*0.5))
	# 	setobject!(vis["circle2"],circle2,
	# 	MeshPhongMaterial(color=RGBA(0,0,0,0.5)))

	setobject!(vis["ball"], HyperSphere(Point3f0(0),
		        convert(Float32,r_ball)),
		        MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

	settransform!(vis["ball"], compose(Translation(0.0,2.0,0.15)))

	hole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.01),convert(Float32,r_ball*1.5))
		setobject!(vis["hole"],hole,
		MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
	settransform!(vis["hole"], compose(Translation(0.0,-2.0,0.0)))

	anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

	T = length(q)
	for t = 1:T
		MeshCat.atframe(anim,t) do
			settransform!(vis["ball"], Translation((q[t][1:3] + [0.0;0.0;0.5*r_ball])...))
		end
	end
	# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
	MeshCat.setanimation!(vis,anim)
end

# Dimensions
nq = 3 # configuration dim
nu_ctrl = 3
nc = 3 # number of contact points
nf = 4 # number of faces for friction cone pyramid
nb = nc*nf

nx = nq
nu = nu_ctrl + nc + nb + nc + nb + 1

idx_u = (1:nu_ctrl)
idx_λ = nu_ctrl .+ (1:nc)
idx_b = nu_ctrl + nc .+ (1:nb)
idx_ψ = nu_ctrl + nc + nb .+ (1:nc)
idx_η = nu_ctrl + nc + nb + nc .+ (1:nb)
idx_s = nu_ctrl + nc + nb + nc + nb + 1

# Parameters
Δt = 0.1  # time step
μ1 = 0.1  # coefficient of friction
μ2 = 0.1  # coefficient of friction
μ3 = 0.1  # coefficient of friction

m = 1.0  # mass

# ellipsoid
ellipsoid(x,y,rx,ry) = (x^2)/(rx^2) + (y^2)/(ry^2) - 1
∇ellipsoid(x,y,rx,ry) = [2.0*x/(rx^2); 2.0*y/(ry^2); 0.0]
get_y(x,rx,ry) = sqrt((1.0 - ((x)^2)/((rx)^2))*(ry^2))

ellipsoid(x,y,xc,yc,rx,ry) = ((x-xc)^2)/(rx^2) + ((y-yc)^2)/(ry^2) - 1
∇ellipsoid(x,y,xc,yc,rx,ry) = [2.0*(x-xc)/(rx^2); 2.0*(y-yc)/(ry^2); 0.0]
get_y(x,xc,yc,rx,ry) = sqrt((1.0 - ((x-xc)^2)/((rx)^2))*(ry^2)) + yc

rx1 = 1.5
ry1 = 1.5

rx2 = 2.5
ry2 = 2.5

a_exp = 3.0
shift_exp = 2.0

α_particle = 1.0

model = GolfParticle(m,μ1,μ2,μ3,
                 nx,nu,nu_ctrl,
                 nc,nf,nb,
                 idx_u,
                 idx_λ,
                 idx_b,
                 idx_ψ,
                 idx_η,
                 idx_s,
                 α_particle,
				 rx1,ry1,
				 rx2,ry2,
				 a_exp,shift_exp)

using Colors
using CoordinateTransformations
using FileIO
using MeshIO
using GeometryTypes
using MeshCat
using Rotations
using Meshing

# vis = Visualizer()
# open(vis)
#
# visualize!(vis,model,[],Δt=Δt,r_ball=0.1)
