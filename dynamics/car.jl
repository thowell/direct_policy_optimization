mutable struct Car
    nx::Int
    nu::Int
end

function dynamics(model::Car,x,u)
    @SVector [u[1]*cos(x[3]), u[1]*sin(x[3]), u[2]]
end

nx,nu = 3,2
model = Car(nx,nu)

function visualize!(vis,p::Car,q; Δt=0.1,r=0.25)

    obj_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/cybertruck/cybertruck.obj")
    mtl_path = joinpath(pwd(),"/home/taylor/Research/direct_policy_optimization/dynamics/cybertruck/cybertruck.mtl")

    ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.05)
    setobject!(vis["cybertruck"],ctm)
    settransform!(vis["cybertruck"], LinearMap(RotZ(pi)*RotX(pi/2.0)))

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q)

        MeshCat.atframe(anim,t) do
            x = [q[t][1];q[t][2];0.0]
            settransform!(vis["cybertruck"], compose(Translation(x),LinearMap(RotZ(q[t][3]+pi)*RotX(pi/2.0))))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
