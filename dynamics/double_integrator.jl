mutable struct DoubleIntegrator
    nx::Int # state dimension
    nu::Int # control dimension
    nw::Int
end

function dynamics(model::DoubleIntegrator,x,u,w)
    @SVector [x[2],u[1]]
    # @SVector [x[3],x[4],u[1],u[2]]
end

nx,nu,nw = 2,1,0
model = DoubleIntegrator(nx,nu,0)

# analytical discrete dynamics

struct DoubleIntegratorAnalytical{T}
    nx::Int
    nu::Int
    nw::Int
    Δt::T
end

function get_dynamics(model::DoubleIntegratorAnalytical)
    nx = model.nx
    nu = model.nu

    Ac = [0.0 1.0; 0.0 0.0]
    Bc = [0.0; 1.0]

    D = exp(model.Δt*[Ac Bc; zeros(1,nx+nu)])
    A = D[1:nx,1:nx]
    B = D[1:nx,nx .+ (1:nu)]

    A, B
end

function discrete_dynamics(model::DoubleIntegratorAnalytical,x⁺,x,u,h,w,t)
    A, B = get_dynamics(model)
    x⁺ - A*x - B*u - w
end
function discrete_dynamics(model::DoubleIntegratorAnalytical,x,u,h,w,t)
    A, B = get_dynamics(model)
    A*x + B*u + w
end

model_analytical = DoubleIntegratorAnalytical(2,1,2,0.1)
