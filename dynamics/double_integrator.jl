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
end

function dynamics(model::DoubleIntegratorHybrid,x,u)
    @SVector [x[2],u[1]]
end

nx_hybrid,nu_hybrid = 2,1
model_hybrid = DoubleIntegratorHybrid(nx_hybrid,nu_hybrid,0)

function Δ(model::DoubleIntegratorHybrid,x)ones(model.nx))
    x + @SVector [1.0, 0.0]
end

function discrete_dynamics(model::DoubleIntegratorHybrid,x⁺,x,u,h,t)
    if t == model.Tm
        Δx = Δ(model,x)
        xm = xm_rk3_implicit(model,x⁺,Δx,u,h)
        x⁺ - Δx - h[1]/6*dynamics(model,Δx,u) - 4*h[1]/6*dynamics(model,xm,u) - h[1]/6*dynamics(model,x⁺,u)
    else
        xm = xm_rk3_implicit(model,x⁺,x,u,h)
        x⁺ - x - h[1]/6*dynamics(model,x,u) - 4*h[1]/6*dynamics(model,xm,u) - h[1]/6*dynamics(model,x⁺,u)
    end
end

function discrete_dynamics(model::DoubleIntegratorHybrid,x,u,h,t)
    if t == model.Tm
        x = Δ(model,x)
    end

    k1 = k2 = k3 = zero(x)
    k1 = h*dynamics(model,x,u)
    k2 = h*dynamics(model,x + 0.5*k1,u)
    k3 = h*dynamics(model,x - k1 + 2.0*k2,u)
    x + (k1 + 4.0*k2 + k3)/6.0
end

mutable struct DoubleIntegrator2D
    nx::Int # state dimension
    nu::Int # control dimension
end

function dynamics(model::DoubleIntegrator2D,x,u)
    # @SVector [x[2],u[1]]
    @SVector [x[3],x[4],u[1],u[2]]
end

model_2D = DoubleIntegrator2D(4,2)
