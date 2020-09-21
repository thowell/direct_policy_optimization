using LinearAlgebra, Plots, ForwardDiff

μ = [0.0; 0.0]
Σ = Diagonal(0.25*ones(2))

mvg(w) = 1.0*exp(-0.5*(w[1:2] - μ)'*inv(Σ)*(w[1:2] - μ))

mvg(μ)
mvg(ones(2))

x = range(-2,stop=2,length=100)
y = range(-2,stop=2,length=100)

μ_test = [[x[i];y[j]] for i = 1:100 for j = 1:100]
pdf = mvg.(μ_test)

plot3d(hcat(μ_test...)[1,:],hcat(μ_test...)[2,:],pdf)
x0 = [0.25;-.5;mvg([0.25;-.5])]
plot3d!([x0[1]],[x0[2]],[mvg(x0[1:2])],marker=:circle,color=:black)
function ϕ(x)
	[x[3] - mvg(x[1:2])]
end

function get_tan_y(x_init)
	t = range(-2.0,stop=2.0,length=10)
	x = [x_init[1] for i = 1:10]
	y = [x_init[2] + t[i] for i = 1:10]
	z = [x_init[3] + ForwardDiff.jacobian(ϕ,x_init)[2]*t[i] for i = 1:10]

	return x,y,z
end

function get_tan_x(x_init)
	t = range(-2.0,stop=2.0,length=10)
	x = [x_init[1] + t[i] for i = 1:10]
	y = [x_init[2] for i = 1:10]
	z = [x_init[3] + ForwardDiff.jacobian(ϕ,x_init)[1]*t[i] for i = 1:10]

	return x,y,z
end

xt,yt,zt = get_tan_y(x0)
xt_,yt_,zt_ = get_tan_x(x0)

plot3d!(xt,yt,zt,marker=:circle,color=:red)


v2 = [x0[1];x0[2];x0[3]]
v3 = vec(ForwardDiff.jacobian(ϕ,x0))

v5 = cross(v2,v3)
cross(v5,v3)
ForwardDiff.jacobian(ϕ,x0)

ForwardDiff.gradient(mvg,x0)

ForwardDiff.derivative(exp,0)

exp(0)-1

plot(range(-1,stop=1,length=100),exp.(3.0*range(-1,stop=1,length=100)))
#
# ellipsoid
ellipsoid(x,y,rx,ry) = (x^2)/(rx^2) + (y^2)/(ry^2) - 1
∇ellipsoid(x,y,rx,ry) = [2.0*x/(rx^2); 2.0*y/(ry^2); 0.0]

get_y(x,rx,ry) = sqrt((1.0 - ((x)^2)/((rx)^2))*(ry^2))
rx1 = 1.5
ry1 = 1.5

rx2 = 2.5
ry2 = 2.5

x1 = range(-rx1,stop=rx1,length=100)
x2 = range(-rx2,stop=rx2,length=100)

plot(x1,get_y.(x1,rx1,ry1),color=:black)
plot!(x2,get_y.(x2,rx2,ry2),color=:red,
	aspect_ratio=:equal)

ellipsoid(x1[1],get_y(x1[1],rx1,ry2),rx1,ry1)
x0 = [2.0; 0.0]
ellipsoid(-1.5,0,rx1,ry1)
ellipsoid(-1.5-0.1,0,rx1,ry1)
-1.0*ellipsoid(-2.5,0,rx2,ry2)
-1.0*ellipsoid(-2.5-0.1,0,rx2,ry2)

y0 = 0.0
tmpx(z) = ellipsoid(z,y0,rx1,rx2)
tmpy(z) = ellipsoid(x1[1],z,rx1,rx2)
ForwardDiff.derivative(tmpx,x1[1]) - 2.0*x1[1]/(rx1^2)
ForwardDiff.derivative(tmpy,y0[1]) - 2.0*y0[1]/(ry1^2)

function ϕ(x)
	[ellipsoid(x[1],x[2],rx1,ry1);
	 -1.0*ellipsoid(x[1],x[2],rx2,ry2)]
end

function N(x)
	[transpose(∇ellipsoid(x[1],x[2],rx1,ry1));
	 -1.0*transpose(∇ellipsoid(x[1],x[2],rx2,ry2))]
end

x0 = [1.0;get_y(1.0,rx1,ry1);0.0]
ϕ(x0)
N(x0)

function P(x)
	z0 = [0.0;0.0;1.0]
	[transpose(z0);
	 transpose(cross(z0,∇ellipsoid(x[1],x[2],rx1,ry1)));
	 transpose(z0);
	 transpose(cross(z0,∇ellipsoid(x[1],x[2],rx2,ry2)))]
end

P(x0)

# exponential surface

a = 1.0
t = range(-3,stop=1,length=100)
plot(t .+ 3,exp.(a.*t),aspect_ratio=:equal)

function ϕ(x)
	[x[3] - exp(a*x[1])]
end

function N(x)
	[-a*exp(a*x[1]) 0.0 1.0]
end

ForwardDiff.jacobian(ϕ,x0) - N(x0)

function P(x)
	y0 = [0.0; 1.0; 0.0]

	[transpose(y0);
	 transpose(cross(y0,[-a*exp(a*x[1]); 0.0; 1.0]))]
end

P(x0)

# visualize

using Colors
using CoordinateTransformations
using FileIO
using MeshIO
using GeometryTypes#:Vec,HyperRectangle,HyperSphere,Point3f0,Cylinder
using MeshCat
using Rotations

vis = Visualizer()
open(vis)

using Meshing
f = x -> x[3] - exp(a*x[1])
sdf = SignedDistanceField(f, HyperRectangle(Vec(-5, -6, -1), Vec(10, 6, 4)))
mesh = HomogenousMesh(sdf, MarchingTetrahedra())
setobject!(vis["slope"], mesh,
           MeshPhongMaterial(color=RGBA{Float32}(86/255, 125/255, 70/255, 1.0)))
settransform!(vis["slope"], compose(Translation(-3.0,3.0,0.0),LinearMap(RotZ(pi/2))))

circle1 = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.5),convert(Float32,rx1))
	setobject!(vis["circle1"],circle1,
	MeshPhongMaterial(color=RGBA(86/255,125/255,20/255,1.0)))

setobject!(vis["ball"], HyperSphere(Point3f0(0),
	        convert(Float32,0.1)),
	        MeshPhongMaterial(color=RGBA(1,1,1,1.0)))

settransform!(vis["ball"], compose(Translation(-2.0,0.0,0.15)))

hole = Cylinder(Point3f0(0,0,0),Point3f0(0,0,0.085),convert(Float32,0.15))
	setobject!(vis["hole"],hole,
	MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
settransform!(vis["hole"], compose(Translation(2.0,0.0,0.0)))
