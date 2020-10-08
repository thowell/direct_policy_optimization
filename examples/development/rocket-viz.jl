using Colors # Handle RGB colors
using CoordinateTransformations # Translations and rotations
using FileIO # Save and load files
using GeometryTypes # Define geometric shape
using LinearAlgebra
using MeshCat # Visualize 3D animations
using MeshIO # Load meshes in MeshCat
using Meshing
include(joinpath(pwd(),"dynamics/visualize.jl"))


vis = Visualizer()
open(vis);

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

anim = MeshCat.Animation(convert(Int,floor(1/dt_sim_sample)))
q = z_sample

for t = 1:length(q)
	MeshCat.atframe(anim,t) do
		settransform!(vis["rocket"], compose(Translation(q[t][1]+rkt_offset[1],0.0+rkt_offset[2],q[t][2]+rkt_offset[3]),LinearMap(RotY(-1.0*q[t][3])*RotZ(pi)*RotX(pi/2.0))))
	end
end
# settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
MeshCat.setanimation!(vis,anim)
