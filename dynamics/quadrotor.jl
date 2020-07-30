include("quaternion.jl")
mutable struct Quadrotor{T}
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
      q = normalize(Quaternion(view(z,4:7)))
      v = view(z,8:10)
      ω = view(z,11:13)

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

      SVector{13}([v; SVector(0.5*(q*Quaternion(zero(x[1]), ω...))); model.g + (1/model.m)*(q*F); model.Jinv*(τ - cross(ω,model.J*ω))])
end

nx = 13
nu = 4
model = Quadrotor(0.5,
                  Diagonal(@SVector[0.0023, 0.0023, 0.004]),
                  Diagonal(@SVector[1.0/0.0023, 1.0/0.0023, 1.0/0.004]),
                  @SVector[0.0,0.0,-9.81],
                  0.175,
                  1.0,
                  0.0245,
                  nx,nu)

# dynamics(model,rand(nx),rand(nu))
