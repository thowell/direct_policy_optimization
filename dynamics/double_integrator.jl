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

function Δ(model::DoubleIntegratorHybrid,x)
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

mutable struct DoubleIntegratorActuated
    nx::Int # state dimension
    nu::Int # control dimension
    nu_original::Int
    Tm::Int
end

function dynamics(model::DoubleIntegratorActuated,x,u)
    @SVector [x[2],u[1]]
    # @SVector [x[3],x[4],u[1],u[2]]
end

nx_actuated,nu_actuated,nu = 2,3,1
model_actuated = DoubleIntegratorActuated(nx_actuated,nu_actuated,nu,0)

function Δ(model::DoubleIntegratorActuated,x)
    x + @SVector [1.0, 0.0]
end

function discrete_dynamics(model::DoubleIntegratorActuated,x⁺,x,u,h,t)
    uo = view(u,1:model.nu_original)
    # uw = view(u,model.nu_original+1:model.nu)

    if t == model.Tm
        Δx = Δ(model,x)
        xm = xm_rk3_implicit(model,x⁺,Δx,uo,h)
        x⁺ - Δx - h[1]/6*dynamics(model,Δx,uo) - 4*h[1]/6*dynamics(model,xm,uo) - h[1]/6*dynamics(model,x⁺,uo)
    else
        xm = xm_rk3_implicit(model,x⁺,x,uo,h)
        x⁺ - x - h[1]/6*dynamics(model,x,uo) - 4*h[1]/6*dynamics(model,xm,uo) - h[1]/6*dynamics(model,x⁺,uo)
    end
end
