mutable struct DoubleIntegrator
    nx::Int # state dimension
    nu::Int # control dimension
end

function dynamics(model::DoubleIntegrator,x,u)
    @SVector [x[2], u[1]]
end

nx,nu = 2,1
model = DoubleIntegrator(nx,nu)
