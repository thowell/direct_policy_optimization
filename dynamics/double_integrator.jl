mutable struct DoubleIntegrator
    nx::Int # state dimension
    nu::Int # control dimension
end

function dynamics(model::DoubleIntegrator,x,u)
    @SVector [x[2],u[1]]
    # @SVector [x[3],x[4],u[1],u[2]]
end

nx,nu = 2,1
model = DoubleIntegrator(nx,nu)

mutable struct DoubleIntegratorHybrid
    nx::Int # state dimension
    nu::Int # control dimension

    Tm::Int
    xm
    transition_type::Symbol
end

function dynamics(model::DoubleIntegratorHybrid,x,u)
    @SVector [x[2],u[1]]
end

nx_hybrid,nu_hybrid = 2,1
model_hybrid = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,0,zeros(nx_hybrid),:event)
