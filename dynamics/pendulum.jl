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

nx,nu = 2,1
model = Pendulum(1.0,0.1,0.5,0.25,9.81,nx,nu)

# Pendulum with free final time
mutable struct PendulumFT{T}
    m::T    # mass
    b::T    # friction
    lc::T   # length to center of mass
    I::T    # inertia
    g::T    # gravity
    nx::Int # state dimension
    nu::Int # control dimension
end

function dynamics(model::PendulumFT,x,u)
    @SVector [x[2],
              u[1]/(model.m*model.lc*model.lc) - model.g*sin(x[1])/model.lc - model.b*x[2]/(model.m*model.lc*model.lc)]
end

function discrete_dynamics(model::PendulumFT,x⁺,x,u,h,t)
    midpoint_implicit(model,x⁺,x,u[1:end-1],u[end])
end

nx_ft,nu_ft = 2,2
model_ft = PendulumFT(1.0,0.1,0.5,0.25,9.81,nx_ft,nu_ft)
