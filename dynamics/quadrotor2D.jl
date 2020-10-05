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

nx,nu,nw = 6,2,0
model = Quadrotor2D(1.0,1.0,1.0,9.81,nx,nu,nw)
