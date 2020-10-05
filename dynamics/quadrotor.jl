using Rotations, StaticArrays, LinearAlgebra

abstract type AbstractQuadrotor end

mutable struct Quadrotor{T} <: AbstractQuadrotor
      m::T
      J
      Jinv
      g
      L::T
      kf::T
      km::T

      nx::Int
      nu::Int
end

function dynamics(model::Quadrotor,z,u)
      # states
      x = view(z,1:3)
      r = view(z,4:6)
      v = view(z,7:9)
      ω = view(z,10:12)

      # controls
      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      # forces
      F1 = model.kf*w1
      F2 = model.kf*w2
      F3 = model.kf*w3
      F4 = model.kf*w4
      F = @SVector [0.0, 0.0, F1+F2+F3+F4] #total rotor force in body frame

      # moments
      M1 = model.km*w1;
      M2 = model.km*w2;
      M3 = model.km*w3;
      M4 = model.km*w4;
      τ = @SVector [model.L*(F2-F4), model.L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      SVector{12}([v; 0.25*((1-r'*r)*ω - 2*cross(ω,r) + 2*(ω'*r)*r); model.g + (1/model.m)*MRP(r[1],r[2],r[3])*F; model.Jinv*(τ - cross(ω,model.J*ω))])
end

nx = 12
nu = 4
model = Quadrotor(0.5,
      Diagonal(@SVector[0.0023, 0.0023, 0.004]),
      Diagonal(@SVector[1.0/0.0023, 1.0/0.0023, 1.0/0.004]),
      @SVector[0.0,0.0,-9.81],
      0.175,
      1.0,
      0.0245,
      nx,nu)

mutable struct QuadrotorFT{T} <: AbstractQuadrotor
      m::T
      J
      Jinv
      g
      L::T
      kf::T
      km::T

      nx::Int
      nu::Int
      nu_ctrl::Int
end

function dynamics(model::QuadrotorFT,z,u)
      # states
      x = view(z,1:3)
      r = view(z,4:6)
      v = view(z,7:9)
      ω = view(z,10:12)

      # controls
      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      # forces
      F1 = model.kf*w1
      F2 = model.kf*w2
      F3 = model.kf*w3
      F4 = model.kf*w4
      F = @SVector [0.0, 0.0, F1+F2+F3+F4] #total rotor force in body frame

      # moments
      M1 = model.km*w1;
      M2 = model.km*w2;
      M3 = model.km*w3;
      M4 = model.km*w4;
      τ = @SVector [model.L*(F2-F4), model.L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      SVector{12}([v; 0.25*((1-r'*r)*ω - 2*cross(ω,r) + 2*(ω'*r)*r); model.g + (1/model.m)*MRP(r[1],r[2],r[3])*F; model.Jinv*(τ - cross(ω,model.J*ω))])
end

function discrete_dynamics(model::QuadrotorFT,x⁺,x,u,h,t)
    midpoint_implicit(model,x⁺,x,u[1:end-1],u[end])
end

nu_ft = 5
model_ft = QuadrotorFT(0.5,
                  Diagonal(@SVector[0.0023, 0.0023, 0.004]),
                  Diagonal(@SVector[1.0/0.0023, 1.0/0.0023, 1.0/0.004]),
                  @SVector[0.0,0.0,-9.81],
                  0.175,
                  1.0,
                  0.0245,
                  nx,nu_ft,nu)

# dynamics(model,rand(nx),rand(nu))
function visualize!(vis,p::AbstractQuadrotor,q; Δt=0.1)

    # obj_path = joinpath(pwd(),"/home/taylor/Research/contact_implicit_trajectory_optimization/models/cybertruck/cybertruck.obj")
    # mtl_path = joinpath(pwd(),"/home/taylor/Research/contact_implicit_trajectory_optimization/models/cybertruck/cybertruck.mtl")
    #
    # ctm = ModifiedMeshFileObject(obj_path,mtl_path,scale=0.05)
    # setobject!(vis["cybertruck"],ctm,MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
    # settransform!(vis["cybertruck"], LinearMap(RotZ(pi)*RotX(pi/2.0)))
    r = model.L
    setobject!(vis["quad"], HyperRectangle(Vec(0,0,0),Vec(2r,2r,2r)))

    anim = MeshCat.Animation(convert(Int,floor(1/Δt)))

    for t = 1:length(q)

        MeshCat.atframe(anim,t) do
            settransform!(vis["quad"], compose(Translation(q[t][1:3]),LinearMap(MRP(q[t][4:6]...))))
        end
    end
    # settransform!(vis["/Cameras/default"], compose(Translation(-1, -1, 0),LinearMap(RotZ(pi/2))))
    MeshCat.setanimation!(vis,anim)
end
