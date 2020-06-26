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

Δt = 0.1

model = DoubleIntegrator1D(1.0) # nominal model
