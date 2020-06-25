# parameterized model
mutable struct DoubleIntegrator1D
    # z = (x,ẋ)
    mx
end

# double integrator dynamics (1D)
n = 2 # number of states
m = 1 # number of controls

function dynamics(model::DoubleIntegrator1D,z,u)
    @SVector [z[2], u[1]/model.mx]
end

# discrete dynamics (midpoint)
Δt = 0.1
function midpoint(model,z,u,Δt)
    z + Δt*dynamics(model,z + 0.5*Δt*dynamics(model,z,u),u)
end

model = DoubleIntegrator1D(1.0) # nominal model
