mutable struct Pendulum{T}
    m::T    # mass
    b::T    # friction
    lc::T   # length to center of mass
    I::T    # inertia
    g::T    # gravity
    nx::Int # state dimension
    nu::Int # control dimension
end

function dynamics(model::Pendulum,x,u)
    @SVector [x[2],
              u[1]/(model.m*model.lc*model.lc) - model.g*sin(x[1])/model.lc - model.b*x[2]/(model.m*model.lc*model.lc)]
end

function kinematics(model::Pendulum,q)
    @SVector [model.lc*sin(q[1]),
              -model.lc*cos(q[1])]
end

nx,nu = 2,1
model = Pendulum(1.0,0.1,0.5,0.25,9.81,nx,nu)
