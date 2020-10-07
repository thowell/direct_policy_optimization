mutable struct Quadrotor2D{T}
    L::T
    m::T
    I::T
    g::T

    nx::Int
    nu::Int
    nw::Int
end

function dynamics(model::Quadrotor2D,x,u,w)
    qdd1 = -sin(x[3])/model.m*(u[1]+u[2])
    qdd2 = -model.g + cos(x[3])/model.m*(u[1]+u[2])
    qdd3 = model.L/model.I*(-u[1]+u[2])
    @SVector [x[4],x[5],x[6],qdd1,qdd2,qdd3]
end

function discrete_dynamics(model::Quadrotor2D,x⁺,x,u,h,w,t)
    midpoint_implicit(model,x⁺,x,u,h,w) + w
end

function discrete_dynamics(model::Quadrotor2D,x,u,h,w,t)
    midpoint(model,x,u,h,w) + w
end

nx,nu,nw = 6,2,6
model = Quadrotor2D(0.5,1.0,1.0/12.0*(2*0.5^2),9.81,nx,nu,nw)
